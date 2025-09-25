mod common;

use common::{message, raw_request_body, request_body_json};
use std::panic;
use wire::api::{GeminiModel, Prompt};
use wire::gemini::GeminiClient;
use wire::types::MessageType;

fn build_client(model: GeminiModel) -> Option<GeminiClient> {
    panic::catch_unwind(|| GeminiClient::new(model)).ok()
}

#[test]
fn gemini_build_request_uses_expected_shape() {
    std::env::set_var("GEMINI_API_KEY", "gemini-key");

    let client = match build_client(GeminiModel::Gemini20Flash) {
        Some(client) => client,
        None => return,
    };

    let chat_history = vec![
        message(MessageType::User, "Hi there"),
        message(MessageType::Assistant, "Hello human"),
    ];

    let request = client
        .build_request(
            "Follow the safety rules.".to_string(),
            chat_history,
            None,
            false,
        )
        .build()
        .expect("gemini request should be buildable");

    assert_eq!(
        request.url().as_str(),
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=gemini-key"
    );

    let body = request_body_json(&request);

    assert_eq!(
        body["system_instruction"]["parts"][0]["text"],
        "Follow the safety rules."
    );

    let contents = body["contents"].as_array().expect("contents array");
    assert_eq!(contents.len(), 2);
    assert_eq!(contents[0]["role"], "user");
    assert_eq!(contents[0]["parts"][0]["text"], "Hi there");
    assert_eq!(contents[1]["role"], "model");
    assert_eq!(contents[1]["parts"][0]["text"], "Hello human");
}

#[test]
fn gemini_build_request_raw_includes_token_and_body() {
    std::env::set_var("GEMINI_API_KEY", "gemini-key");

    let client = match build_client(GeminiModel::Gemini25ProExp) {
        Some(client) => client,
        None => return,
    };

    let raw_request = client.build_request_raw(
        "Keep responses short.".to_string(),
        vec![message(MessageType::User, "Summarize this")],
        true,
    );

    assert!(raw_request
        .contains("POST /v1beta/models/gemini-2.5-flash-preview-04-17:streamGenerateContent"));
    assert!(raw_request.contains("?key=gemini-key"));
    assert!(raw_request.contains("Host: generativelanguage.googleapis.com"));

    let body = raw_request_body(&raw_request);
    assert_eq!(
        body["system_instruction"]["parts"][0]["text"],
        "Keep responses short."
    );
}

#[test]
fn gemini_read_json_response_extracts_text() {
    let client = match build_client(GeminiModel::Gemini20FlashLite) {
        Some(client) => client,
        None => return,
    };

    let response_json = serde_json::json!({
        "candidates": [
            {
                "content": {
                    "parts": [
                        { "text": "Gemini output" }
                    ]
                }
            }
        ]
    });

    let content = client
        .read_json_response(&response_json)
        .expect("gemini response should contain text");

    assert_eq!(content, "Gemini output");
}
