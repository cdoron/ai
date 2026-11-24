// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional coverage for the two-step `consult_advisor` managed tool
//! (`anthropic/consult-advisor.yaml`).
//!
//! An executor model call to the `ConsultAdvisor` tool transitions the
//! `iterative_request_router` to a second step that consults a separate
//! advisor backend, then folds the advisor's text answer back into the
//! executor's conversation as a `tool_result` before transitioning back.
//! This exercises the full round trip plus the two failure modes fixed in
//! review: a tool_use block the filter cannot answer alongside the managed
//! call, and a failing advisor backend.

use std::collections::HashMap;

use praxis_test_utils::{
    StatefulCapturingBackend, build_pipeline, example_config_path, free_port, http_send, json_post, parse_body,
    parse_status, patch_yaml, start_proxy,
};
use serde_json::{Value, json};

const EXAMPLE: &str = "anthropic/consult-advisor.yaml";

/// Read the example and rewrite the proxy, executor-backend, and
/// advisor-backend ports to the given local stubs.
fn load_config(proxy_port: u16, executor_port: u16, advisor_port: u16) -> praxis_core::config::Config {
    let yaml = std::fs::read_to_string(example_config_path(EXAMPLE)).expect("read consult-advisor example");
    let yaml = patch_yaml(
        &yaml,
        proxy_port,
        &HashMap::from([("127.0.0.1:8000", executor_port), ("127.0.0.1:8001", advisor_port)]),
    );
    praxis_core::config::Config::from_yaml(&yaml).expect("parse consult-advisor example")
}

/// The native Anthropic Messages client request declaring the
/// `ConsultAdvisor` tool.
fn client_request() -> Value {
    json!({
        "model": "executor-model",
        "max_tokens": 1024,
        "messages": [{
            "role": "user",
            "content": [{"type": "text", "text": "Ask the advisor whether it is safe to run rm -rf / in production."}],
        }],
        "tools": [{
            "name": "ConsultAdvisor",
            "description": "Ask a second expert model for advice before answering a risky question.",
            "input_schema": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]},
        }],
    })
}

/// An executor round that calls the managed `ConsultAdvisor` tool.
fn executor_tool_call_round(tool_use_id: &str, question: &str) -> String {
    json!({
        "id": "msg_round0",
        "type": "message",
        "role": "assistant",
        "model": "executor-model",
        "content": [{"type": "tool_use", "id": tool_use_id, "name": "ConsultAdvisor", "input": {"question": question}}],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    })
    .to_string()
}

/// A terminal executor round that answers in plain text.
fn executor_answer_round(text: &str) -> String {
    json!({
        "id": "msg_round1",
        "type": "message",
        "role": "assistant",
        "model": "executor-model",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 20, "output_tokens": 10},
    })
    .to_string()
}

/// A terminal advisor round that answers in plain text.
fn advisor_answer_round(text: &str) -> String {
    json!({
        "id": "msg_advisor",
        "type": "message",
        "role": "assistant",
        "model": "advisor-model",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 8, "output_tokens": 12},
    })
    .to_string()
}

#[test]
fn consult_advisor_example_builds() {
    let config = load_config(free_port(), 19_701, 19_702);
    let _pipeline = build_pipeline(&config);
}

#[test]
fn consult_advisor_round_trip_folds_advisor_answer_into_executor_response() {
    let executor = StatefulCapturingBackend::new(vec![
        (
            200,
            executor_tool_call_round("toolu_1", "Is it safe to run rm -rf / in production?"),
        ),
        (
            200,
            executor_answer_round("The advisor said: no, that would destroy the filesystem."),
        ),
    ])
    .start_with_shutdown();
    let advisor = StatefulCapturingBackend::new(vec![(
        200,
        advisor_answer_round("No, that would destroy the filesystem."),
    )])
    .start_with_shutdown();
    let proxy_port = free_port();
    let config = load_config(proxy_port, executor.port(), advisor.port());
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &json_post("/v1/messages", &client_request().to_string()));

    assert_eq!(parse_status(&raw), 200, "round trip should succeed: {raw}");
    let body: Value = serde_json::from_str(&parse_body(&raw)).expect("parse response body");
    assert_eq!(
        body["content"][0]["text"], "The advisor said: no, that would destroy the filesystem.",
        "the client receives the executor's post-consultation answer: {body}"
    );

    let advisor_requests = advisor.requests();
    assert_eq!(advisor_requests.len(), 1, "exactly one advisor callout");
    let advisor_body: Value = serde_json::from_str(&advisor_requests[0].body).expect("parse advisor request body");
    let advisor_text = advisor_body["messages"][0]["content"][0]["text"]
        .as_str()
        .expect("advisor user text");
    assert!(
        advisor_text.contains("Is it safe to run rm -rf / in production?"),
        "the advisor receives the tool's question: {advisor_body}"
    );
    assert!(
        advisor_text.contains("Ask the advisor whether it is safe to run rm -rf / in production."),
        "the advisor also receives the client's original conversation, not just the tool's question: {advisor_body}"
    );
    assert!(
        advisor_body.get("system").and_then(Value::as_str).is_some(),
        "the advisor request carries a system prompt: {advisor_body}"
    );

    let executor_requests = executor.requests();
    assert_eq!(
        executor_requests.len(),
        2,
        "two executor rounds: the tool call, then the fold-back"
    );
    let second_round: Value = serde_json::from_str(&executor_requests[1].body).expect("parse second executor round");
    let messages = second_round["messages"].as_array().expect("messages array");
    let tool_result = messages
        .last()
        .expect("a tool_result turn follows the retained conversation");
    assert_eq!(
        tool_result["content"][0]["tool_use_id"], "toolu_1",
        "the fold-back answers the pending tool_use id: {tool_result}"
    );
    assert_eq!(
        tool_result["content"][0]["content"], "No, that would destroy the filesystem.",
        "the fold-back carries the advisor's answer text: {tool_result}"
    );
}

#[test]
fn a_tool_use_block_alongside_the_managed_call_is_rejected() {
    let unanswerable_round = json!({
        "id": "msg_round0",
        "type": "message",
        "role": "assistant",
        "model": "executor-model",
        "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "ConsultAdvisor", "input": {"question": "safe?"}},
            {"type": "tool_use", "id": "toolu_2", "name": "SomeOtherTool", "input": {}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    })
    .to_string();
    let executor = StatefulCapturingBackend::new(vec![(200, unanswerable_round)]).start_with_shutdown();
    let advisor = StatefulCapturingBackend::new(vec![(200, advisor_answer_round("unused"))]).start_with_shutdown();
    let proxy_port = free_port();
    let config = load_config(proxy_port, executor.port(), advisor.port());
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &json_post("/v1/messages", &client_request().to_string()));

    assert_eq!(
        parse_status(&raw),
        502,
        "a tool_use block the filter cannot answer is rejected, not silently dropped: {raw}"
    );
    assert_eq!(
        advisor.requests().len(),
        0,
        "the advisor is never consulted for an unanswerable round"
    );
}

#[test]
fn a_failing_advisor_backend_does_not_leak_its_raw_response_to_the_client() {
    let executor =
        StatefulCapturingBackend::new(vec![(200, executor_tool_call_round("toolu_1", "safe?"))]).start_with_shutdown();
    let advisor = StatefulCapturingBackend::new(vec![(
        500,
        json!({"internal": "advisor backend secret detail"}).to_string(),
    )])
    .start_with_shutdown();
    let proxy_port = free_port();
    let config = load_config(proxy_port, executor.port(), advisor.port());
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &json_post("/v1/messages", &client_request().to_string()));

    assert_eq!(
        parse_status(&raw),
        502,
        "an advisor failure surfaces as a clean gateway error: {raw}"
    );
    assert!(
        !parse_body(&raw).contains("advisor backend secret detail"),
        "the advisor's raw response body must not reach the client: {raw}"
    );
}
