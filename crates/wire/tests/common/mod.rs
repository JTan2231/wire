#![allow(dead_code)]

pub mod mock_server;

use wire::api::{OpenAIModel, API};
use wire::types::{Function, FunctionCall, Message, MessageType, Tool, ToolWrapper};

pub fn message(message_type: MessageType, content: &str) -> Message {
    Message {
        message_type,
        content: content.to_string(),
        api: default_api(),
        system_prompt: String::new(),
        tool_calls: None,
        tool_call_id: None,
        name: None,
        input_tokens: 0,
        output_tokens: 0,
    }
}

fn default_api() -> API {
    API::OpenAI(OpenAIModel::GPT4o)
}

pub fn function_call(id: &str, name: &str, arguments: serde_json::Value) -> FunctionCall {
    FunctionCall {
        id: id.to_string(),
        call_type: "function".to_string(),
        function: Function {
            name: name.to_string(),
            arguments: arguments.to_string(),
        },
    }
}

pub fn sample_tool(name: &str) -> Tool {
    Tool {
        function_type: "function".to_string(),
        name: name.to_string(),
        description: "example tool".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {},
        }),
        function: Box::new(ToolWrapper(|args| args)),
    }
}

pub fn request_body_json(request: &reqwest::Request) -> serde_json::Value {
    let bytes = request
        .body()
        .and_then(|body| body.as_bytes())
        .expect("request body should be JSON bytes");

    serde_json::from_slice(bytes).expect("request body should deserialize")
}

pub fn raw_request_body(raw: &str) -> serde_json::Value {
    let idx = raw
        .rfind("\r\n\r\n")
        .expect("raw request should contain header terminator");

    let body = raw[idx + 4..].trim();

    serde_json::from_str(body).expect("raw request body should deserialize")
}
