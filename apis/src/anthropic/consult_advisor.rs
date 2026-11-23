// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! A two-step server-owned tool for the Anthropic Messages agentic loop:
//! when the executor model calls a managed tool, Praxis consults a second
//! ("advisor") model as a genuine second inference backend, folding its
//! answer back into the executor's conversation as a `tool_result`.
//!
//! Modeled on [`super::AnthropicWebSearchFilter`] and this crate's
//! `where_is_treasure` filter, but spanning two named
//! `iterative_request_router` steps instead of one: one filter instance
//! with `role: executor` in the step that runs the main model, and a
//! second instance with `role: advisor` in a step that runs the advisor
//! model. `IterationState.accumulator` carries the pending tool-call id
//! and the executor's assistant turn across the step transition and back.
//!
//! As with `where_is_treasure`, the executor's own rounds are forced to
//! the buffered subrequest transport regardless of the client's
//! requested `stream` value, and the terminal (non-looping) answer is
//! re-encoded as a single Anthropic Messages SSE lifecycle when the
//! client asked to stream. The advisor's own round is never directly
//! client-visible (it always transitions back to the executor), so it
//! needs none of that.

use async_trait::async_trait;
use bytes::Bytes;
use http::header::{ACCEPT_ENCODING, CONTENT_TYPE, HeaderValue};
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, IterationState, NextIterationBody,
    Rejection, SubRequestResponseMode, parse_filter_config,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::wire;

/// Registry name and filter-results namespace.
const FILTER_NAME: &str = "consult_advisor";
/// IRR action that transitions the executor step to the advisor step.
const ACTION_CONSULT: &str = "consult";
/// IRR action that transitions the advisor step back to the executor step.
const ACTION_FOLD_BACK: &str = "fold_back";
/// IRR action that returns the current response to the client.
const ACTION_DONE: &str = "done";
/// Accumulator entry holding the latest serialized executor request.
const REQUEST_ACCUMULATOR_KEY: &str = "consult_advisor.request";
/// Accumulator entry holding the pending tool call across the advisor hop.
const PENDING_ACCUMULATOR_KEY: &str = "consult_advisor.pending";
/// Default tool name advertised to and matched against the client.
const DEFAULT_TOOL_NAME: &str = "ConsultAdvisor";
/// Default `max_tokens` for the advisor's one-shot request.
const DEFAULT_ADVISOR_MAX_TOKENS: u32 = 1024;

/// Default maximum body size (1 MiB).
fn default_max_body_bytes() -> usize {
    1_048_576
}

/// Default `max_tokens` for the advisor's one-shot request.
fn default_advisor_max_tokens() -> u32 {
    DEFAULT_ADVISOR_MAX_TOKENS
}

/// Which half of the two-step flow a `ConsultAdvisorFilter` instance plays.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Role {
    /// Runs in the step that talks to the main (executor) model: detects
    /// the managed tool call, and folds the advisor's eventual answer
    /// back into the executor's conversation.
    Executor,
    /// Runs in the step that talks to the advisor model: builds the
    /// one-shot advisor request and always transitions back.
    Advisor,
}

/// YAML configuration for [`ConsultAdvisorFilter`].
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConsultAdvisorConfig {
    /// Which half of the flow this instance plays.
    role: Role,
    /// Tool name advertised to and matched against the client. Defaults to
    /// [`DEFAULT_TOOL_NAME`]. Used by both roles.
    tool_name: Option<String>,
    /// Model name sent in the advisor's own request body. Required for
    /// `role: advisor`.
    target_model: Option<String>,
    /// Optional system prompt for the advisor's one-shot request.
    /// Only meaningful for `role: advisor`.
    system_prompt: Option<String>,
    /// `max_tokens` for the advisor's one-shot request. Only meaningful
    /// for `role: advisor`.
    #[serde(default = "default_advisor_max_tokens")]
    advisor_max_tokens: u32,
    /// Maximum request/response body size in bytes for `StreamBuffer` mode.
    #[serde(default = "default_max_body_bytes")]
    max_body_bytes: usize,
}

/// The tool call pending while the advisor step runs, carried across the
/// step transition in `IterationState.accumulator`.
#[derive(Serialize, Deserialize)]
struct PendingConsult {
    /// Anthropic tool-use identifier the eventual `tool_result` answers.
    tool_use_id: String,
    /// The executor assistant turn's complete content array, preserved so
    /// any text preceding the tool call survives in conversation history.
    assistant_content: Vec<Value>,
}

/// Executes a server-owned managed tool call by consulting a second
/// inference backend, folding its answer back as a `tool_result`.
pub struct ConsultAdvisorFilter {
    /// Which half of the flow this instance plays.
    role: Role,
    /// Tool name advertised to and matched against the client.
    tool_name: String,
    /// Model name sent in the advisor's own request body.
    target_model: Option<String>,
    /// Optional system prompt for the advisor's one-shot request.
    system_prompt: Option<String>,
    /// `max_tokens` for the advisor's one-shot request.
    advisor_max_tokens: u32,
    /// Maximum request/response body size in bytes for `StreamBuffer` mode.
    max_body_bytes: usize,
}

impl ConsultAdvisorFilter {
    /// Build the filter from its YAML configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when `config` does not match
    /// [`ConsultAdvisorConfig`], or when `role: advisor` is missing
    /// `target_model`.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: ConsultAdvisorConfig = parse_filter_config(FILTER_NAME, config)?;
        if cfg.role == Role::Advisor && cfg.target_model.is_none() {
            return Err(FilterError::from(format!(
                "{FILTER_NAME}: 'target_model' is required for role: advisor"
            )));
        }
        Ok(Box::new(Self {
            role: cfg.role,
            tool_name: cfg.tool_name.unwrap_or_else(|| DEFAULT_TOOL_NAME.to_owned()),
            target_model: cfg.target_model,
            system_prompt: cfg.system_prompt,
            advisor_max_tokens: cfg.advisor_max_tokens,
            max_body_bytes: cfg.max_body_bytes,
        }))
    }

    /// Whether this request body is an IRR re-entry (a later agentic
    /// round), distinguished by the router-owned [`IterationState`]
    /// carrying the previous round's response. A fresh, first-round
    /// request has none.
    fn is_reentry(ctx: &HttpFilterContext<'_>) -> bool {
        ctx.extensions
            .get::<IterationState>()
            .and_then(|state| state.previous_response.as_ref())
            .is_some()
    }

    /// Force the upcoming backend round to the buffered subrequest
    /// transport, regardless of what the client asked for.
    fn force_buffered_transport(ctx: &mut HttpFilterContext<'_>) {
        ctx.set_subrequest_response_mode(SubRequestResponseMode::Buffered);
    }

    /// Whether the original client request asked for a streamed response.
    fn original_wanted_streaming(ctx: &HttpFilterContext<'_>) -> bool {
        ctx.extensions
            .get::<IterationState>()
            .and_then(|state| serde_json::from_slice::<Value>(&state.original_request.body).ok())
            .and_then(|value| value.get("stream").and_then(Value::as_bool))
            .unwrap_or(false)
    }

    /// Force `"stream": false` on a request body, then serialize it.
    fn force_non_streaming_and_serialize(mut request: Value) -> Result<Bytes, FilterError> {
        if let Some(object) = request.as_object_mut() {
            object.insert("stream".to_owned(), Value::Bool(false));
        }
        serde_json::to_vec(&request)
            .map(Bytes::from)
            .map_err(|error| FilterError::from(format!("{FILTER_NAME}: request serialization failed: {error}")))
    }

    /// Find a managed tool-use block anywhere in an assistant message's
    /// content and return its `id`, the question from `input.question`,
    /// and the complete content array (so any preceding text blocks are
    /// preserved verbatim in the next round's conversation history).
    fn managed_call_from_response(&self, bytes: &[u8]) -> Option<(String, String, Vec<Value>)> {
        let response: Value = serde_json::from_slice(bytes).ok()?;
        if response.get("type").and_then(Value::as_str) != Some("message") {
            return None;
        }
        if response.get("role").and_then(Value::as_str) != Some("assistant") {
            return None;
        }
        let content = response.get("content")?.as_array()?;
        let tool_use = content.iter().find(|block| {
            block.get("type").and_then(Value::as_str) == Some("tool_use")
                && block.get("name").and_then(Value::as_str) == Some(self.tool_name.as_str())
        })?;
        let id = tool_use.get("id").and_then(Value::as_str)?;
        if id.is_empty() {
            return None;
        }
        let question = tool_use
            .get("input")
            .and_then(|input| input.get("question"))
            .and_then(Value::as_str)?;
        Some((id.to_owned(), question.to_owned(), content.clone()))
    }

    /// Concatenate every `text` content block in a buffered Anthropic
    /// Messages response body.
    fn response_text(bytes: &[u8]) -> Option<String> {
        let response: Value = serde_json::from_slice(bytes).ok()?;
        Some(concat_text_blocks(response.get("content").and_then(Value::as_array)))
    }

    // -- executor role --------------------------------------------------

    /// Round 0 (or any non-reentry round) of the executor step: force
    /// non-streaming transport and continue unmodified otherwise.
    fn handle_executor_fresh_request(
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
    ) -> Result<FilterAction, FilterError> {
        let Some(bytes) = body.as_deref() else {
            return Ok(FilterAction::Continue);
        };
        let Ok(request) = serde_json::from_slice::<Value>(bytes) else {
            return Ok(FilterAction::Continue);
        };
        *body = Some(Self::force_non_streaming_and_serialize(request)?);
        Self::force_buffered_transport(ctx);
        Ok(FilterAction::Continue)
    }

    /// Executor re-entry: fold the advisor's answer (or the previous
    /// executor round's response, if nothing is pending) into the
    /// retained conversation and continue the executor model.
    fn handle_executor_reentry(
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
    ) -> Result<FilterAction, FilterError> {
        let Some(iteration_state) = ctx.extensions.get::<IterationState>() else {
            return Err(FilterError::from(format!(
                "{FILTER_NAME}: IRR iteration state unavailable during re-entry"
            )));
        };
        let (pending, advisor_answer) = Self::recover_pending_and_answer(iteration_state)?;

        let request_bytes = iteration_state
            .accumulator
            .get(REQUEST_ACCUMULATOR_KEY)
            .unwrap_or(&iteration_state.original_request.body);
        let mut request: Value = serde_json::from_slice(request_bytes)
            .map_err(|error| FilterError::from(format!("{FILTER_NAME}: retained request parsing failed: {error}")))?;
        if let Err(rejection) = append_consult_turns(&mut request, &pending, &advisor_answer) {
            return Ok(FilterAction::Reject(rejection));
        }
        Self::commit_executor_request(ctx, body, request)
    }

    /// Recover the pending tool call and the advisor's answer text,
    /// stashed/produced across the advisor step transition.
    fn recover_pending_and_answer(iteration_state: &IterationState) -> Result<(PendingConsult, String), FilterError> {
        let Some(previous_response) = iteration_state.previous_response.as_ref() else {
            return Err(FilterError::from(format!(
                "{FILTER_NAME}: previous IRR response unavailable during re-entry"
            )));
        };
        let Some(pending_bytes) = iteration_state.accumulator.get(PENDING_ACCUMULATOR_KEY) else {
            return Err(FilterError::from(format!(
                "{FILTER_NAME}: no pending consult-advisor call during executor re-entry"
            )));
        };
        let pending: PendingConsult = serde_json::from_slice(pending_bytes)
            .map_err(|error| FilterError::from(format!("{FILTER_NAME}: pending consult state corrupt: {error}")))?;
        let Some(advisor_answer) = Self::response_text(&previous_response.body) else {
            return Err(FilterError::from(format!(
                "{FILTER_NAME}: advisor response was not valid JSON during fold-back"
            )));
        };
        Ok((pending, advisor_answer))
    }

    /// Force non-streaming transport, serialize the rebuilt executor
    /// request, and retain it for the IRR's next round.
    fn commit_executor_request(
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        request: Value,
    ) -> Result<FilterAction, FilterError> {
        let rebuilt = Self::force_non_streaming_and_serialize(request)?;
        Self::force_buffered_transport(ctx);
        let iteration_state = ctx.extensions.get_mut::<IterationState>().ok_or_else(|| {
            FilterError::from(format!(
                "{FILTER_NAME}: IRR iteration state unavailable while retaining request"
            ))
        })?;
        iteration_state.accumulator.remove(PENDING_ACCUMULATOR_KEY);
        iteration_state
            .accumulator
            .insert(REQUEST_ACCUMULATOR_KEY.to_owned(), rebuilt.clone());
        ctx.extensions.insert(NextIterationBody(rebuilt.clone()));
        ctx.request_headers_to_set
            .push((CONTENT_TYPE, HeaderValue::from_static("application/json")));
        *body = Some(rebuilt);
        Ok(FilterAction::Continue)
    }

    // -- advisor role -----------------------------------------------------

    /// Advisor step re-entry (always true; this step is never round 0):
    /// recover the pending call from the executor's response, stash it
    /// for the eventual fold-back, and build the advisor's own one-shot
    /// request.
    fn handle_advisor_dispatch(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
    ) -> Result<FilterAction, FilterError> {
        let Some(iteration_state) = ctx.extensions.get::<IterationState>() else {
            return Err(FilterError::from(format!(
                "{FILTER_NAME}: IRR iteration state unavailable while dispatching to the advisor"
            )));
        };
        let Some(previous_response) = iteration_state.previous_response.as_ref() else {
            return Err(FilterError::from(format!(
                "{FILTER_NAME}: previous IRR response unavailable while dispatching to the advisor"
            )));
        };
        let Some((tool_use_id, question, assistant_content)) = self.managed_call_from_response(&previous_response.body)
        else {
            return Err(FilterError::from(format!(
                "{FILTER_NAME}: executor response no longer contains a managed {} call",
                self.tool_name
            )));
        };
        let pending = PendingConsult {
            tool_use_id,
            assistant_content,
        };
        let rebuilt = self.build_advisor_request(&question)?;
        Self::force_buffered_transport(ctx);
        Self::commit_advisor_dispatch(ctx, body, &pending, rebuilt)
    }

    /// Build the advisor's one-shot request body for `question`.
    fn build_advisor_request(&self, question: &str) -> Result<Bytes, FilterError> {
        let Some(target_model) = self.target_model.as_deref() else {
            return Err(FilterError::from(format!(
                "{FILTER_NAME}: 'target_model' missing for role: advisor"
            )));
        };
        let mut advisor_request = json!({
            "model": target_model,
            "max_tokens": self.advisor_max_tokens,
            "stream": false,
            "messages": [{"role": "user", "content": [{"type": "text", "text": question}]}],
        });
        if let Some(system_prompt) = &self.system_prompt
            && let Some(object) = advisor_request.as_object_mut()
        {
            object.insert("system".to_owned(), Value::String(system_prompt.clone()));
        }
        let rebuilt = serde_json::to_vec(&advisor_request).map_err(|error| {
            FilterError::from(format!("{FILTER_NAME}: advisor request serialization failed: {error}"))
        })?;
        Ok(Bytes::from(rebuilt))
    }

    /// Stash the pending call and set the advisor request as this round's
    /// outgoing body.
    fn commit_advisor_dispatch(
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        pending: &PendingConsult,
        rebuilt: Bytes,
    ) -> Result<FilterAction, FilterError> {
        let pending_bytes = serde_json::to_vec(&pending).map_err(|error| {
            FilterError::from(format!("{FILTER_NAME}: pending consult serialization failed: {error}"))
        })?;
        let iteration_state = ctx.extensions.get_mut::<IterationState>().ok_or_else(|| {
            FilterError::from(format!(
                "{FILTER_NAME}: IRR iteration state unavailable while dispatching to the advisor"
            ))
        })?;
        iteration_state
            .accumulator
            .insert(PENDING_ACCUMULATOR_KEY.to_owned(), Bytes::from(pending_bytes));
        ctx.extensions.insert(NextIterationBody(rebuilt.clone()));
        ctx.request_headers_to_set
            .push((CONTENT_TYPE, HeaderValue::from_static("application/json")));
        *body = Some(rebuilt);
        Ok(FilterAction::Continue)
    }

    // -- terminal SSE re-encoding (executor role only) -------------------

    /// Re-encode a buffered Anthropic Messages JSON response as a single
    /// Anthropic Messages SSE lifecycle, for a client that asked to stream
    /// but whose backend round was forced to buffered transport.
    fn buffered_to_sse(bytes: &[u8]) -> Option<Vec<u8>> {
        let response: Value = serde_json::from_slice(bytes).ok()?;
        Some(render_sse_lifecycle(&TerminalAnswer::from_response(&response)))
    }
}

/// Fields of a buffered Messages response needed to re-encode it as SSE.
struct TerminalAnswer {
    /// Message ID, echoed into `message_start`.
    id: String,
    /// Model name, echoed into `message_start`.
    model: String,
    /// Stop reason, echoed into `message_delta`.
    stop_reason: String,
    /// Usage object, echoed into `message_delta` verbatim.
    usage: Value,
    /// Concatenated text of every `text` content block.
    text: String,
}

impl TerminalAnswer {
    /// Extract the fields needed to re-encode `response` as SSE.
    fn from_response(response: &Value) -> Self {
        let text = concat_text_blocks(response.get("content").and_then(Value::as_array));
        Self {
            id: response
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or("msg_consult_advisor")
                .to_owned(),
            model: response.get("model").and_then(Value::as_str).unwrap_or("").to_owned(),
            stop_reason: response
                .get("stop_reason")
                .and_then(Value::as_str)
                .unwrap_or("end_turn")
                .to_owned(),
            usage: response.get("usage").cloned().unwrap_or_else(|| json!({})),
            text,
        }
    }
}

/// Concatenate every `text` content block, treating an absent content
/// array as having none.
fn concat_text_blocks(content: Option<&Vec<Value>>) -> String {
    content
        .into_iter()
        .flatten()
        .filter(|block| block.get("type").and_then(Value::as_str) == Some("text"))
        .filter_map(|block| block.get("text").and_then(Value::as_str))
        .collect()
}

/// Append the executor's assistant tool call and the advisor-answer user
/// turn to the retained executor request.
fn append_consult_turns(request: &mut Value, pending: &PendingConsult, advisor_answer: &str) -> Result<(), Rejection> {
    let Some(messages) = request.get_mut("messages").and_then(Value::as_array_mut) else {
        return Err(wire::error_rejection(
            400,
            wire::ErrorType::InvalidRequest,
            "messages must be an array for consult-advisor re-entry",
        ));
    };
    let mut assistant_turn = serde_json::Map::new();
    assistant_turn.insert("role".to_owned(), Value::String("assistant".to_owned()));
    assistant_turn.insert("content".to_owned(), Value::Array(pending.assistant_content.clone()));
    messages.push(Value::Object(assistant_turn));
    messages.push(json!({
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": pending.tool_use_id, "content": advisor_answer}],
    }));
    if request.get("tool_choice").is_some()
        && let Some(object) = request.as_object_mut()
    {
        object.insert("tool_choice".to_owned(), json!({"type": "auto"}));
    }
    Ok(())
}

/// Render one coherent Anthropic Messages SSE lifecycle for a terminal,
/// non-looping answer.
fn render_sse_lifecycle(answer: &TerminalAnswer) -> Vec<u8> {
    let mut out = Vec::new();
    push_opening_events(&mut out, answer);
    push_closing_events(&mut out, answer);
    out
}

/// Push `message_start` and `content_block_start`.
fn push_opening_events(out: &mut Vec<u8>, answer: &TerminalAnswer) {
    push_event(
        out,
        "message_start",
        &json!({"type": "message_start", "message": {
            "id": answer.id, "type": "message", "role": "assistant", "model": answer.model,
            "content": [], "stop_reason": null, "stop_sequence": null,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }}),
    );
    push_event(
        out,
        "content_block_start",
        &json!({"type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "citations": null, "text": ""}}),
    );
}

/// Push `content_block_delta`, `content_block_stop`, `message_delta`, and
/// `message_stop`.
fn push_closing_events(out: &mut Vec<u8>, answer: &TerminalAnswer) {
    push_event(
        out,
        "content_block_delta",
        &json!({"type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": answer.text}}),
    );
    push_event(
        out,
        "content_block_stop",
        &json!({"type": "content_block_stop", "index": 0}),
    );
    push_event(
        out,
        "message_delta",
        &json!({"type": "message_delta",
            "delta": {"stop_reason": answer.stop_reason, "stop_sequence": null}, "usage": answer.usage}),
    );
    push_event(out, "message_stop", &json!({"type": "message_stop"}));
}

/// Serialize one Anthropic Messages SSE frame: `event: <name>\ndata:
/// <json>\n\n`.
fn push_event(out: &mut Vec<u8>, event: &str, data: &Value) {
    out.extend_from_slice(b"event: ");
    out.extend_from_slice(event.as_bytes());
    out.extend_from_slice(b"\ndata: ");
    out.extend_from_slice(&serde_json::to_vec(data).unwrap_or_default());
    out.extend_from_slice(b"\n\n");
}

/// Publish the loop decision for the IRR transition table.
fn set_action(ctx: &mut HttpFilterContext<'_>, action: &'static str) -> Result<(), FilterError> {
    ctx.filter_results
        .entry(FILTER_NAME)
        .or_default()
        .set("action", action)?;
    Ok(())
}

/// Whether the upstream response succeeded, treating an absent header
/// (not yet populated) as success so classification proceeds.
fn is_success_response(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.response_header
        .as_ref()
        .is_none_or(|response| response.status.is_success())
}

#[async_trait]
impl HttpFilter for ConsultAdvisorFilter {
    fn name(&self) -> &'static str {
        FILTER_NAME
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadWrite(praxis_filter::BodyRequirement::Optional)
    }

    fn request_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(self.max_body_bytes),
        }
    }

    fn response_body_access(&self) -> BodyAccess {
        BodyAccess::ReadWrite(praxis_filter::BodyRequirement::Optional)
    }

    fn response_body_mode(&self) -> BodyMode {
        // Every round is forced to the buffered subrequest transport, so
        // there is no live stream to compose with; StreamBuffer hands this
        // filter the complete body in one call.
        BodyMode::StreamBuffer {
            max_bytes: Some(self.max_body_bytes),
        }
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        // A compressed body cannot be parsed as JSON.
        ctx.request_headers_to_remove.push(ACCEPT_ENCODING);
        Ok(FilterAction::Continue)
    }

    async fn on_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }
        match self.role {
            Role::Executor => {
                if Self::is_reentry(ctx) {
                    Self::handle_executor_reentry(ctx, body)
                } else {
                    Self::handle_executor_fresh_request(ctx, body)
                }
            },
            Role::Advisor => self.handle_advisor_dispatch(ctx, body),
        }
    }

    async fn on_response(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        // Only the executor role's terminal round is ever externally
        // visible (the IRR discards every looping/transitioning round
        // before it reaches the client), so unconditionally overriding
        // Content-Type here on every executor round is safe and correctly
        // applies to whichever round ends up terminal. The advisor's own
        // round is never externally visible at all.
        if self.role == Role::Executor
            && Self::original_wanted_streaming(ctx)
            && let Some(response) = ctx.response_header.as_mut()
        {
            response
                .headers
                .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        }
        Ok(FilterAction::Continue)
    }

    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }
        if !is_success_response(ctx) {
            set_action(ctx, ACTION_DONE)?;
            return Ok(FilterAction::Continue);
        }
        match self.role {
            Role::Executor => {
                let Some(bytes) = body.as_deref() else {
                    set_action(ctx, ACTION_DONE)?;
                    return Ok(FilterAction::Continue);
                };
                if self.managed_call_from_response(bytes).is_some() {
                    set_action(ctx, ACTION_CONSULT)?;
                    return Ok(FilterAction::Continue);
                }
                set_action(ctx, ACTION_DONE)?;
                if Self::original_wanted_streaming(ctx)
                    && let Some(sse) = Self::buffered_to_sse(bytes)
                {
                    *body = Some(Bytes::from(sse));
                }
                Ok(FilterAction::Continue)
            },
            Role::Advisor => {
                set_action(ctx, ACTION_FOLD_BACK)?;
                Ok(FilterAction::Continue)
            },
        }
    }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, reason = "tests")]
mod tests {
    use super::*;

    fn executor_filter() -> ConsultAdvisorFilter {
        ConsultAdvisorFilter {
            role: Role::Executor,
            tool_name: DEFAULT_TOOL_NAME.to_owned(),
            target_model: None,
            system_prompt: None,
            advisor_max_tokens: DEFAULT_ADVISOR_MAX_TOKENS,
            max_body_bytes: default_max_body_bytes(),
        }
    }

    #[test]
    fn from_config_rejects_advisor_role_without_target_model() {
        let yaml = serde_yaml::from_str("role: advisor").unwrap();
        let Err(error) = ConsultAdvisorFilter::from_config(&yaml) else {
            panic!("expected an error when target_model is missing for role: advisor");
        };
        assert!(error.to_string().contains("target_model"));
    }

    #[test]
    fn finds_managed_call_with_question_among_other_content() {
        let body = json!({
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me ask for advice."},
                {"type": "tool_use", "id": "toolu_1", "name": "ConsultAdvisor", "input": {"question": "Is this safe?"}},
            ],
        });
        let bytes = serde_json::to_vec(&body).unwrap();
        let (id, question, content) = executor_filter()
            .managed_call_from_response(&bytes)
            .expect("managed call");
        assert_eq!(id, "toolu_1");
        assert_eq!(question, "Is this safe?");
        assert_eq!(content.len(), 2);
    }

    #[test]
    fn ignores_unmanaged_tool_name() {
        let body = json!({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_1", "name": "SomeOtherTool", "input": {"question": "?"}}],
        });
        let bytes = serde_json::to_vec(&body).unwrap();
        assert!(executor_filter().managed_call_from_response(&bytes).is_none());
    }

    #[test]
    fn response_text_concatenates_text_blocks() {
        let body = json!({"content": [{"type": "text", "text": "Yes, "}, {"type": "text", "text": "it is safe."}]});
        let bytes = serde_json::to_vec(&body).unwrap();
        assert_eq!(ConsultAdvisorFilter::response_text(&bytes).unwrap(), "Yes, it is safe.");
    }

    #[test]
    fn append_consult_turns_adds_assistant_and_tool_result() {
        let mut request = json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]});
        let pending = PendingConsult {
            tool_use_id: "toolu_1".to_owned(),
            assistant_content: vec![
                json!({"type": "tool_use", "id": "toolu_1", "name": "ConsultAdvisor", "input": {"question": "?"}}),
            ],
        };
        append_consult_turns(&mut request, &pending, "Yes, it is safe.").unwrap();
        let messages = request.get("messages").and_then(Value::as_array).unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1].get("role").and_then(Value::as_str), Some("assistant"));
        let tool_result = &messages[2]["content"][0];
        assert_eq!(tool_result.get("tool_use_id").and_then(Value::as_str), Some("toolu_1"));
        assert_eq!(
            tool_result.get("content").and_then(Value::as_str),
            Some("Yes, it is safe.")
        );
    }

    #[test]
    fn buffered_to_sse_carries_full_text_and_stop_reason() {
        let body = json!({
            "id": "msg_1", "model": "test-model", "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 7},
            "content": [{"type": "text", "text": "Yes, it is safe."}],
        });
        let bytes = serde_json::to_vec(&body).unwrap();
        let sse = ConsultAdvisorFilter::buffered_to_sse(&bytes).expect("sse bytes");
        let text = String::from_utf8(sse).unwrap();
        assert!(text.starts_with("event: message_start\n"));
        assert!(text.contains("Yes, it is safe."));
        assert!(text.ends_with("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"));
    }
}
