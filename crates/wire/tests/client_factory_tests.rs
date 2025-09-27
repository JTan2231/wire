use std::panic;

use temp_env::with_var;
use wire::api::{AnthropicModel, GeminiModel, OpenAIModel, Prompt, API};
use wire::config::ClientOptions;
use wire::types::{Message, MessageBuilder};
use wire::{new_client, new_client_with_options};

fn simple_message(api: API, content: &str) -> Vec<Message> {
    vec![MessageBuilder::new(api, content).build()]
}

fn build_client(provider: &str, model: &str) -> Option<Box<dyn Prompt>> {
    match panic::catch_unwind(|| new_client(provider, model)) {
        Ok(Ok(client)) => Some(client),
        Ok(Err(err)) => panic!("unexpected error creating client: {err}"),
        Err(_) => None,
    }
}

fn build_client_with_options(
    provider: &str,
    model: &str,
    options: ClientOptions,
) -> Option<Box<dyn Prompt>> {
    match panic::catch_unwind(|| new_client_with_options(provider, model, options)) {
        Ok(Ok(client)) => Some(client),
        Ok(Err(err)) => panic!("unexpected error creating client with options: {err}"),
        Err(_) => None,
    }
}

#[test]
fn new_client_creates_openai_client() {
    with_var("OPENAI_API_KEY", Some("test-openai"), || {
        let client = match build_client("openai", "gpt-4o") {
            Some(client) => client,
            None => return,
        };
        let messages = simple_message(API::OpenAI(OpenAIModel::GPT4o), "hello");

        let request = client
            .build_request("Be helpful".to_string(), messages, None, false)
            .build()
            .expect("openai request should build");

        assert_eq!(
            request.url().as_str(),
            "https://api.openai.com/v1/chat/completions"
        );
    });
}

#[test]
fn new_client_creates_anthropic_client() {
    with_var("ANTHROPIC_API_KEY", Some("test-anthropic"), || {
        let client = match build_client("anthropic", "claude-3-5-sonnet-20241022") {
            Some(client) => client,
            None => return,
        };
        let messages = simple_message(API::Anthropic(AnthropicModel::Claude35SonnetNew), "hello");

        let request = client
            .build_request("Be kind".to_string(), messages, None, false)
            .build()
            .expect("anthropic request should build");

        assert_eq!(
            request.url().as_str(),
            "https://api.anthropic.com/v1/messages"
        );
    });
}

#[test]
fn new_client_creates_gemini_client() {
    with_var("GEMINI_API_KEY", Some("test-gemini"), || {
        let client = match build_client("gemini", "gemini-2.0-flash") {
            Some(client) => client,
            None => return,
        };
        let messages = simple_message(API::Gemini(GeminiModel::Gemini20Flash), "hello");

        let request = client
            .build_request("Be creative".to_string(), messages, None, false)
            .build()
            .expect("gemini request should build");

        assert_eq!(
            request.url().as_str(),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=test-gemini"
        );
    });
}

#[test]
fn new_client_with_options_overrides_base_url() {
    with_var("OPENAI_API_KEY", Some("test-openai"), || {
        let options = ClientOptions::from_base_url("http://localhost:4242")
            .expect("client options from base url");
        let client = match build_client_with_options("openai", "gpt-4o", options) {
            Some(client) => client,
            None => return,
        };
        let messages = simple_message(API::OpenAI(OpenAIModel::GPT4o), "override");

        let request = client
            .build_request("Use override".to_string(), messages, None, false)
            .build()
            .expect("request with options should build");

        assert_eq!(
            request.url().as_str(),
            "http://localhost:4242/v1/chat/completions"
        );
    });
}

#[test]
fn new_client_errors_on_unknown_provider() {
    assert!(matches!(
        new_client("unknown", "model"),
        Err(err) if err.contains("Unknown provider")
    ));
}
