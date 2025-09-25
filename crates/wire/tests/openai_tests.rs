mod common;

use common::{function_call, message, raw_request_body, request_body_json, sample_tool};
use std::panic;
use wire::api::{OpenAIModel, Prompt};
use wire::openai::OpenAIClient;
use wire::types::MessageType;

fn build_client(model: OpenAIModel) -> Option<OpenAIClient> {
    panic::catch_unwind(|| OpenAIClient::new(model)).ok()
}

#[test]
fn openai_build_request_includes_system_and_tooling() {
    std::env::set_var("OPENAI_API_KEY", "openai-key");

    let client = match build_client(OpenAIModel::GPT4oMini) {
        Some(client) => client,
        None => return,
    };

    let mut tool_call_message = message(MessageType::FunctionCall, "");
    tool_call_message.tool_calls = Some(vec![function_call(
        "call-1",
        "lookup_weather",
        serde_json::json!({ "zip": "10001" }),
    )]);

    let mut tool_result_message = message(MessageType::FunctionCallOutput, "snow");
    tool_result_message.tool_call_id = Some("call-1".to_string());

    let chat_history = vec![
        message(MessageType::User, "What's the weather?"),
        tool_call_message,
        tool_result_message,
    ];

    let request = client
        .build_request(
            "Always explain your reasoning.".to_string(),
            chat_history,
            Some(vec![sample_tool("lookup_weather")]),
            false,
        )
        .build()
        .expect("openai request should be buildable");

    assert_eq!(
        request.url().as_str(),
        "https://api.openai.com/v1/chat/completions"
    );
    assert_eq!(
        request
            .headers()
            .get(reqwest::header::AUTHORIZATION)
            .expect("auth header present")
            .to_str()
            .unwrap(),
        "Bearer openai-key"
    );

    let body = request_body_json(&request);

    assert_eq!(body["model"], "gpt-4o-mini");
    assert_eq!(body["stream"], false);

    let messages = body["messages"].as_array().expect("messages array");
    assert_eq!(messages.len(), 4);

    assert_eq!(messages[0]["role"], "system");
    assert_eq!(messages[0]["content"], "Always explain your reasoning.");
    assert_eq!(messages[1]["role"], "user");
    assert_eq!(messages[1]["content"], "What's the weather?");

    assert_eq!(messages[2]["role"], "assistant");
    assert!(messages[2]["tool_calls"].is_array());
    assert_eq!(messages[2]["tool_calls"][0]["id"], "call-1");

    assert_eq!(messages[3]["role"], "tool");
    assert_eq!(messages[3]["tool_call_id"], "call-1");

    let tools = body["tools"].as_array().expect("tools array");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["type"], "function");
    assert_eq!(tools[0]["function"]["name"], "lookup_weather");
}

#[test]
fn openai_build_request_adds_reasoning_effort_for_gpt5() {
    std::env::set_var("OPENAI_API_KEY", "openai-key");

    let client = match build_client(OpenAIModel::GPT5) {
        Some(client) => client,
        None => return,
    };

    let request = client
        .build_request(
            "Stay focused.".to_string(),
            vec![message(MessageType::User, "Solve this")],
            None,
            false,
        )
        .build()
        .expect("gpt-5 request should be buildable");

    let body = request_body_json(&request);

    assert_eq!(body["model"], "gpt-5");
    assert_eq!(body["reasoning_effort"], "minimal");
}

#[test]
fn openai_build_request_raw_contains_headers_and_body() {
    std::env::set_var("OPENAI_API_KEY", "openai-key");

    let client = match build_client(OpenAIModel::GPT4o) {
        Some(client) => client,
        None => return,
    };

    let raw = client.build_request_raw(
        "Be concise.".to_string(),
        vec![message(MessageType::User, "Explain quantum physics")],
        true,
    );

    assert!(raw.contains("Authorization: Bearer openai-key"));
    assert!(raw.contains("Content-Type: application/json"));

    let body = raw_request_body(&raw);
    assert_eq!(body["stream"], true);
    assert_eq!(body["model"], "gpt-4o");
}

#[test]
fn openai_read_json_response_extracts_text() {
    let client = match build_client(OpenAIModel::GPT4oMini) {
        Some(client) => client,
        None => return,
    };

    let response_json = serde_json::json!({
        "choices": [
            {
                "message": {
                    "content": "OpenAI reply"
                }
            }
        ]
    });

    let content = client
        .read_json_response(&response_json)
        .expect("openai response should contain text");

    assert_eq!(content, "OpenAI reply");
}
