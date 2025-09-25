mod common;

use common::mock_server::{MockJsonResponse, MockLLMServer, MockResponse, MockRoute};
use common::{function_call, message, request_body_json, sample_tool};
use std::panic;
use temp_env::with_var;
use wire::anthropic::AnthropicClient;
use wire::api::{AnthropicModel, Prompt};
use wire::config::ClientOptions;
use wire::types::MessageType;

fn build_client(model: AnthropicModel) -> Option<AnthropicClient> {
    panic::catch_unwind(|| AnthropicClient::new(model)).ok()
}

#[test]
fn anthropic_build_request_formats_messages_and_tools() {
    std::env::set_var("ANTHROPIC_API_KEY", "anthropic-key");

    let client = match build_client(AnthropicModel::Claude35SonnetNew) {
        Some(client) => client,
        None => return,
    };

    let mut assistant = message(MessageType::Assistant, "");
    assistant.tool_calls = Some(vec![function_call(
        "call-1",
        "lookup_weather",
        serde_json::json!({ "location": "NYC" }),
    )]);

    let tool_result_content = serde_json::json!({ "forecast": "snow" }).to_string();
    let mut tool_result = message(MessageType::FunctionCallOutput, &tool_result_content);
    tool_result.tool_call_id = Some("call-1".to_string());

    let chat_history = vec![
        message(MessageType::User, "What's the weather?"),
        assistant,
        tool_result,
    ];

    let request = client
        .build_request(
            "You are a helpful assistant.".to_string(),
            chat_history,
            Some(vec![sample_tool("lookup_weather")]),
            false,
        )
        .build()
        .expect("request should be buildable");

    assert_eq!(
        request.url().as_str(),
        "https://api.anthropic.com/v1/messages"
    );
    assert_eq!(
        request
            .headers()
            .get("x-api-key")
            .expect("anthropic auth header present")
            .to_str()
            .unwrap(),
        "anthropic-key"
    );

    let body = request_body_json(&request);

    assert_eq!(body["system"], "You are a helpful assistant.");
    assert_eq!(body["model"], "claude-3-5-sonnet-20241022");

    let messages = body["messages"].as_array().expect("messages array");
    assert_eq!(messages.len(), 3);

    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[0]["content"], "What's the weather?");

    let assistant_content = messages[1]["content"]
        .as_array()
        .expect("assistant content");
    assert_eq!(assistant_content[0]["type"], "tool_use");
    assert_eq!(assistant_content[0]["name"], "lookup_weather");
    assert_eq!(
        assistant_content[0]["input"],
        serde_json::json!({ "location": "NYC" })
    );

    let tool_result_content = messages[2]["content"]
        .as_array()
        .expect("tool result content");
    assert_eq!(tool_result_content[0]["type"], "tool_result");
    assert_eq!(tool_result_content[0]["tool_use_id"], "call-1");
    assert_eq!(
        tool_result_content[0]["content"],
        serde_json::json!({ "forecast": "snow" }).to_string()
    );

    let tools = body["tools"].as_array().expect("tools array");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["name"], "lookup_weather");
    assert!(tools[0]["input_schema"].is_object());
}

#[test]
fn anthropic_read_json_response_extracts_text() {
    let client = match build_client(AnthropicModel::Claude35SonnetNew) {
        Some(client) => client,
        None => return,
    };

    let response_json = serde_json::json!({
        "content": [
            {
                "type": "text",
                "text": "Response payload"
            }
        ]
    });

    let content = client
        .read_json_response(&response_json)
        .expect("anthropic response should contain text");

    assert_eq!(content, "Response payload");
}

#[test]
fn anthropic_prompt_integration_uses_mock_server() {
    if std::env::var("WIRE_RUN_MOCK_SERVER_TESTS").is_err() {
        eprintln!("skipping anthropic integration test");
        return;
    }

    with_var("ANTHROPIC_API_KEY", Some("mock-anthropic-key"), || {
        let runtime = tokio::runtime::Runtime::new().expect("runtime for anthropic test");

        runtime.block_on(async {
            let server = MockLLMServer::start(vec![MockRoute::single(
                "/v1/messages",
                MockResponse::Json(MockJsonResponse::new(serde_json::json!({
                    "content": [
                        {
                            "type": "text",
                            "text": "anthropic reply"
                        }
                    ]
                }))),
            )])
            .await
            .expect("mock server starts");

            let options =
                ClientOptions::for_mock_server(&server).expect("client options for mock server");
            let client = AnthropicClient::with_options(AnthropicModel::Claude35SonnetNew, options);

            let response = client
                .prompt(
                    "Assist kindly.".to_string(),
                    vec![message(MessageType::User, "Hello?")],
                )
                .await
                .expect("prompt returns content");

            assert_eq!(response.content, "anthropic reply");

            let recorded = server.requests_for("/v1/messages").await;
            assert_eq!(recorded.len(), 1);

            let headers = &recorded[0].headers;
            assert_eq!(
                headers.get("x-api-key"),
                Some(&"mock-anthropic-key".to_string())
            );
            assert_eq!(
                headers.get("anthropic-version"),
                Some(&"2023-06-01".to_string())
            );

            let payload: serde_json::Value =
                serde_json::from_str(&recorded[0].body_as_string().expect("request body is utf-8"))
                    .expect("request body parses as json");

            assert_eq!(payload["system"], "Assist kindly.");
            assert_eq!(payload["messages"][0]["role"], "user");
            assert_eq!(payload["messages"][0]["content"], "Hello?");

            server.shutdown().await;
        });
    });
}
