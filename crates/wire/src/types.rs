use serde::{Deserialize, Serialize};

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

#[derive(Clone, Debug, Eq, Hash, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "provider", content = "model")]
pub enum API {
    #[serde(rename = "openai")]
    OpenAI(OpenAIModel),
    #[serde(rename = "anthropic")]
    Anthropic(AnthropicModel),
    #[serde(rename = "gemini")]
    Gemini(GeminiModel),
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum OpenAIModel {
    #[serde(rename = "gpt-5")]
    GPT5,
    #[serde(rename = "gpt-4.1")]
    GPT4o,
    #[serde(rename = "gpt-4o-mini")]
    GPT4oMini,
    #[serde(rename = "o1-preview")]
    O1Preview,
    #[serde(rename = "o1-mini")]
    O1Mini,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum AnthropicModel {
    #[serde(rename = "claude-opus-4-1-20250805")]
    ClaudeOpus41,
    #[serde(rename = "claude-opus-4-20250514")]
    ClaudeOpus4,
    #[serde(rename = "claude-sonnet-4-20250514")]
    ClaudeSonnet4,
    #[serde(rename = "claude-3-7-sonnet-20250219")]
    Claude37Sonnet,
    #[serde(rename = "claude-3-5-sonnet-20241022")]
    Claude35SonnetNew,
    #[serde(rename = "claude-3-5-haiku-20241022")]
    Claude35Haiku,
    #[serde(rename = "claude-3-5-sonnet-20240620")]
    Claude35SonnetOld,
    #[serde(rename = "claude-3-haiku-20240307")]
    Claude3Haiku,
    #[serde(rename = "claude-3-opus-20240229")]
    Claude3Opus,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum GeminiModel {
    #[serde(rename = "gemini-2.5-flash-preview-04-17")]
    Gemini25ProExp,
    #[serde(rename = "gemini-2.0-flash")]
    Gemini20Flash,
    #[serde(rename = "gemini-2.0-flash-lite")]
    Gemini20FlashLite,
    #[serde(rename = "gemini-embedding-exp")]
    GeminiEmbedding,
}

impl API {
    pub fn from_strings(provider: &str, model: &str) -> Result<Self, String> {
        match provider {
            "openai" => {
                let model = match model {
                    "gpt-5" => OpenAIModel::GPT5,
                    "gpt-4o" => OpenAIModel::GPT4o,
                    "gpt-4o-mini" => OpenAIModel::GPT4oMini,
                    "o1-preview" => OpenAIModel::O1Preview,
                    "o1-mini" => OpenAIModel::O1Mini,
                    _ => return Err(format!("Unknown OpenAI model: {}", model)),
                };
                Ok(API::OpenAI(model))
            }
            "anthropic" => {
                let model = match model {
                    "claude-opus-4-1-20250805" => AnthropicModel::ClaudeOpus41,
                    "claude-opus-4-20250514" => AnthropicModel::ClaudeOpus4,
                    "claude-sonnet-4-20250514" => AnthropicModel::ClaudeSonnet4,
                    "claude-3-7-sonnet-20250219" => AnthropicModel::Claude37Sonnet,
                    "claude-3-5-sonnet-20241022" => AnthropicModel::Claude35SonnetNew,
                    "claude-3-5-haiku-20241022" => AnthropicModel::Claude35Haiku,
                    "claude-3-5-sonnet-20240620" => AnthropicModel::Claude35SonnetOld,
                    "claude-3-haiku-20240307" => AnthropicModel::Claude3Haiku,
                    "claude-3-opus-20240229" => AnthropicModel::Claude3Opus,
                    _ => return Err(format!("Unknown Anthropic model: {}", model)),
                };
                Ok(API::Anthropic(model))
            }
            "gemini" => {
                let model = match model {
                    "gemini-2.5-flash-preview-04-17" => GeminiModel::Gemini25ProExp,
                    "gemini-2.0-flash" => GeminiModel::Gemini20Flash,
                    "gemini-2.0-flash-lite" => GeminiModel::Gemini20FlashLite,
                    "gemini-embedding-exp" => GeminiModel::GeminiEmbedding,
                    _ => return Err(format!("Unknown Gemini model: {}", model)),
                };
                Ok(API::Gemini(model))
            }
            _ => Err(format!("Unknown provider: {}", provider)),
        }
    }

    pub fn to_strings(&self) -> (String, String) {
        match self {
            API::OpenAI(model) => {
                let model_str = match model {
                    OpenAIModel::GPT5 => "gpt-5",
                    OpenAIModel::GPT4o => "gpt-4o",
                    OpenAIModel::GPT4oMini => "gpt-4o-mini",
                    OpenAIModel::O1Preview => "o1-preview",
                    OpenAIModel::O1Mini => "o1-mini",
                };
                ("openai".to_string(), model_str.to_string())
            }
            API::Anthropic(model) => {
                let model_str = match model {
                    AnthropicModel::ClaudeOpus41 => "claude-opus-4-1-20250805",
                    AnthropicModel::ClaudeOpus4 => "claude-opus-4-20250514",
                    AnthropicModel::ClaudeSonnet4 => "claude-sonnet-4-20250514",
                    AnthropicModel::Claude37Sonnet => "claude-3-7-sonnet-20250219",
                    AnthropicModel::Claude35SonnetNew => "claude-3-5-sonnet-20241022",
                    AnthropicModel::Claude35Haiku => "claude-3-5-haiku-20241022",
                    AnthropicModel::Claude35SonnetOld => "claude-3-5-sonnet-20240620",
                    AnthropicModel::Claude3Haiku => "claude-3-haiku-20240307",
                    AnthropicModel::Claude3Opus => "claude-3-opus-20240229",
                };
                ("anthropic".to_string(), model_str.to_string())
            }
            API::Gemini(model) => {
                let model_str = match model {
                    GeminiModel::Gemini25ProExp => "gemini-2.5-flash-preview-04-17",
                    GeminiModel::Gemini20Flash => "gemini-2.0-flash",
                    GeminiModel::Gemini20FlashLite => "gemini-2.0-flash-lite",
                    GeminiModel::GeminiEmbedding => "gemini-embedding-exp",
                };
                ("gemini".to_string(), model_str.to_string())
            }
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
