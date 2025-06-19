use proc_macro::TokenStream;
use quote::{ToTokens, format_ident, quote};
use syn::{FnArg, ItemFn, Type, parse_macro_input};

// TODO: This _needs_ rectified with the vizier macro because this doesn't work and that one does
#[proc_macro]
pub fn get_tool_from_function(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input with syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated)
        .into_iter()
        .collect::<Vec<_>>();

    if input.len() != 2 {
        panic!("Expected function name and description");
    }

    let func = &input[0];
    let description = if let syn::Expr::Lit(lit) = &input[1] {
        if let syn::Lit::Str(s) = &lit.lit {
            s.value()
        } else {
            panic!("Expected string literal for description");
        }
    } else {
        panic!("Expected string literal for description");
    };

    let func_name = if let syn::Expr::Path(path) = func {
        path.path.segments.last().unwrap().ident.to_string()
    } else {
        panic!("Expected function name");
    };

    let mut properties = std::collections::HashMap::new();

    let func_name_str = func_name.to_string();
    let func_item: ItemFn =
        syn::parse_str(&format!("{}", quote! { #func })).expect("Failed to parse function");

    for arg in func_item.sig.inputs.iter() {
        println!("CHECK 2");
        if let FnArg::Typed(pat_type) = arg {
            if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                let arg_name = pat_ident.ident.to_string();
                println!("arg name: {}", arg_name);
                let arg_type = match &*pat_type.ty {
                    Type::Path(type_path) => {
                        let type_name = type_path.path.segments.last().unwrap().ident.to_string();
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

    // Convert the properties to a string representation
    let parameters_json = serde_json::to_string(&serde_json::json!({
        "type": "object",
        "properties": properties
    }))
    .unwrap();

    let wrapper_name = format_ident!("{}_wrapper", func_name);

    quote! {
        Tool {
            function_type: "function".to_string(),
            name: #func_name.to_string(),
            description: #description.to_string(),
            parameters: serde_json::from_str(#parameters_json).unwrap(),
            function: Box::new(ToolWrapper(#wrapper_name)),
        }
    }
    .into()
}
