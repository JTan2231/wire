use proc_macro::TokenStream;
use quote::{ToTokens, quote};
use syn::{FnArg, ItemFn, Type, parse_macro_input};

#[proc_macro]
pub fn get_tool_from_function(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::ExprCall);

    let func = &input.func;
    let args = &input.args;

    let func_name = if let syn::Expr::Path(path) = &**func {
        path.path.segments.last().unwrap().ident.to_string()
    } else {
        panic!("Expected function name");
    };

    let description = if let syn::Expr::Lit(lit) = &args[0] {
        if let syn::Lit::Str(s) = &lit.lit {
            s.value()
        } else {
            panic!("Expected string literal for description");
        }
    } else {
        panic!("Expected string literal for description");
    };

    let mut properties = std::collections::HashMap::new();

    if let Ok(func_item) = syn::parse::<ItemFn>(func.to_token_stream().into()) {
        for arg in func_item.sig.inputs.iter() {
            if let FnArg::Typed(pat_type) = arg {
                if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                    let arg_name = pat_ident.ident.to_string();
                    let arg_type = match &*pat_type.ty {
                        Type::Path(type_path) => {
                            let type_name =
                                type_path.path.segments.last().unwrap().ident.to_string();
                            match type_name.as_str() {
                                "String" => "string",
                                "i32" | "i64" => "integer",
                                "f32" | "f64" => "number",
                                "bool" => "boolean",
                                _ => "object",
                            }
                        }
                        _ => "object",
                    };
                    properties.insert(arg_name, arg_type);
                }
            }
        }
    }

    // Convert the properties to a string representation
    let parameters_json = serde_json::to_string(&serde_json::json!({
        "type": "object",
        "properties": properties
    }))
    .unwrap();

    quote! {
        Tool {
            function_type: "function".to_string(),
            name: #func_name.to_string(),
            description: #description.to_string(),
            parameters: serde_json::from_str(#parameters_json).unwrap(),
        }
    }
    .into()
}
