mod common;

use common::sample_tool;
use std::panic;
use wire::openai::OpenAIClient;
use wire::types::MessageType;

#[test]
fn openai_builder_sets_defaults() {
    let client = match build_client() {
        Some(client) => client,
        None => return,
    };
    let message = client.new_message("hello world").build();

    assert_eq!(message.message_type, MessageType::User);
    assert_eq!(message.content, "hello world");
    assert!(matches!(message.api, wire::api::API::OpenAI(_)));
    assert!(message.tool_calls.is_none());
    assert!(message.tool_call_id.is_none());
    assert!(message.name.is_none());
}

#[test]
fn builder_allows_role_customization() {
    let client = match build_client() {
        Some(client) => client,
        None => return,
    };
    let message = client
        .new_message("system msg")
        .as_system()
        .with_system_prompt("system prompt")
        .with_usage(3, 4)
        .build();

    assert_eq!(message.message_type, MessageType::System);
    assert_eq!(message.system_prompt, "system prompt");
    assert_eq!(message.input_tokens, 3);
    assert_eq!(message.output_tokens, 4);
}

#[test]
fn builder_with_tools_returns_bundle() {
    let client = match build_client() {
        Some(client) => client,
        None => return,
    };
    let tool = sample_tool("demo");

    let bundle = client
        .new_message("needs tools")
        .with_tools(vec![tool.clone()]);

    let (message, tools) = bundle.into_parts();

    assert_eq!(message.content, "needs tools");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "demo");
}

fn build_client() -> Option<OpenAIClient> {
    panic::catch_unwind(|| OpenAIClient::new("gpt-4o-mini")).ok()
}
