use koopa::ir::Program;
use crate::front::{ast::*, ir::*};
use crate::CompilerError;
use crate::utils::SourceMap;

pub mod ast;
pub mod ir;

pub fn generate_ir(ast: &CompUnit, source_map: Option<SourceMap>) -> Result<Program, CompilerError> {
    let mut ctx = IrContext::new(source_map);
    ast.generate_ir(&mut ctx)?;
    Ok(ctx.program)
}