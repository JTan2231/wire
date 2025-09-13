use serde::{Deserialize, Serialize};

use crate::API;

#[derive(PartialEq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum MessageType {
    System,
    User,
    Assistant,
    FunctionCall,
    FunctionCallOutput,
}

// TODO: Refactor types for the Responses API instead of the completions API

impl MessageType {
    pub fn to_string(&self) -> String {
        match self {
            MessageType::System => "system".to_string(),
            MessageType::User => "user".to_string(),
            MessageType::Assistant => "assistant".to_string(),
            MessageType::FunctionCall => "function".to_string(),
            MessageType::FunctionCallOutput => "tool".to_string(),
        }
    }
}

// NOTE: This is only to be used to refer to rust functions
// NOTE: Functions used as tools _must_ have a `fn f(args: serde_json::Value) -> serde_json::Value`
//       type signature
// TODO: This should probably be refactored at some point to keep the functions separated
//       from the struct
#[derive(Debug, Clone, Serialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub function_type: String,
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
    #[serde(skip)]
    pub function: Box<dyn ToolFunction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: Function,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    pub arguments: String,
}

// TODO: Hideous type. Move the tool stuff out of here.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Message {
    // TODO: This gets mapped to `role` in `build_request` and should be more clearly named
    pub message_type: MessageType,

    #[serde(skip_serializing_if = "String::is_empty")]
    pub content: String,
    pub api: API,

    // TODO: Do we really need this with _every_ message?
    pub system_prompt: String,

    // Tool calls made by the model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<FunctionCall>>,

    // Tool call results--actual result content will be in `content` if this isn't None
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    // TODO: These two should probably be somewhere else

    // _Not_ cumulative--per message
    #[serde(skip)]
    pub input_tokens: usize,
    #[serde(skip)]
    pub output_tokens: usize,
}

pub trait ToolFunction: Send + Sync {
    fn call(&self, args: serde_json::Value) -> serde_json::Value;
    fn clone_box(&self) -> Box<dyn ToolFunction>;
    fn debug_fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
}

impl Clone for Box<dyn ToolFunction> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

impl std::fmt::Debug for Box<dyn ToolFunction> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.debug_fmt(f)
    }
}

pub struct ToolWrapper<F>(pub F);

impl<F: Clone> ToolFunction for ToolWrapper<F>
where
    F: Fn(serde_json::Value) -> serde_json::Value + Send + Sync + 'static,
{
    fn call(&self, args: serde_json::Value) -> serde_json::Value {
        self.0(args)
    }

    fn clone_box(&self) -> Box<dyn ToolFunction> {
        Box::new(Self(self.0.clone()))
    }

    fn debug_fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FnWrapper")
    }
}

#[derive(Clone, Debug)]
pub struct RequestParams {
    pub provider: String,
    pub host: String,
    pub path: String,
    pub port: u16,
    pub messages: Vec<Message>,
    pub model: String,
    pub stream: bool,
    pub authorization_token: String,
    pub max_tokens: Option<u16>,
    pub system_prompt: Option<String>,
    pub tools: Option<Vec<Tool>>,
}
