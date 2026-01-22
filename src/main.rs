#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(non_snake_case)]

use lalrpop_util::{lalrpop_mod, ParseError};
use lalrpop_util::lexer::Token;
use std::fs::read_to_string;
use koopa::back::KoopaGenerator;
use koopa::ir::Type;

mod front;
mod back;
mod utils;

use utils::args::Params;
use utils::logger::print_error_and_exit;
use utils::SourceMap;
use crate::front::generate_ir;
use crate::back::generate_asm;

// 重新导出 CompilerError 以便其他模块使用
pub use utils::CompilerError;

// 引用 lalrpop 生成的解析器
// 因为我们刚刚创建了 sysy.lalrpop, 所以模块名是 sysy
lalrpop_mod!(pub sysy);

fn main() {
    if let Err(e) = run() {
        print_error_and_exit(&e, 1);
    }
}

fn format_expected(expected: Vec<String>) -> String {
    if expected.is_empty() {
        "no expected tokens".to_string()
    } else {
        expected.join(", ")
    }
}

fn format_parse_error(source_map: &SourceMap, err: ParseError<usize, Token<'_>, &str>) -> String {
    match err {
        ParseError::InvalidToken { location } => {
            format!("Invalid token at {}", source_map.format_location(location))
        }
        ParseError::UnrecognizedEOF { location, expected } => {
            let expected = format_expected(expected);
            format!(
                "Unexpected end of file at {}. Expected: {}",
                source_map.format_location(location),
                expected
            )
        }
        ParseError::UnrecognizedToken { token, expected } => {
            let (start, tok, _end) = token;
            let expected = format_expected(expected);
            format!(
                "Unrecognized token {:?} at {}. Expected: {}",
                tok,
                source_map.format_location(start),
                expected
            )
        }
        ParseError::ExtraToken { token } => {
            let (start, tok, _end) = token;
            format!(
                "Extra token {:?} at {}",
                tok,
                source_map.format_location(start)
            )
        }
        ParseError::User { error } => {
            format!("Parse error: {}", error)
        }
    }
}

fn run() -> Result<(), CompilerError> {
    // 解析命令行参数
    let params = Params::from_args()?;

    // 目标为 riscv32，设置指针大小为 4 字节
    if params.riscv || params.perf {
        Type::set_ptr_size(4);
    }

    // 读取输入文件
    let input = read_to_string(&params.input)?;
    let source_map = SourceMap::new(&input);

    // 调用 lalrpop 生成的 parser 解析输入文件
    let ast = sysy::CompUnitParser::new()
        .parse(&input)
        .map_err(|e| CompilerError::ParseError(format_parse_error(&source_map, e)))?;

    // println!("{:#?}", ast);

    let program = generate_ir(&ast, Some(source_map))?;

    if params.koopa {
        KoopaGenerator::from_path(&params.output)
            .map_err(|e| CompilerError::CodeGenerationError(format!("Failed to create KoopaGenerator: {}", e)))?
            .generate_on(&program)
            .map_err(|e| CompilerError::CodeGenerationError(format!("Failed to generate Koopa IR: {}", e)))?;
        return Ok(());
    }
    if params.riscv || params.perf {
        let asm = generate_asm(&program)?;
        std::fs::write(&params.output, asm)?;
        return Ok(());
    }
    println!("{:#?}", ast);
    Ok(())
}
