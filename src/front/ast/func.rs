use crate::front::ast::Decl;

use super::{Block, Ident, Span, DataType, Expr};

#[derive(Debug, Clone, PartialEq)]
pub struct FuncFParam {
    pub ty: DataType,
    pub name: String,
    pub is_array: bool,
    pub dims: Vec<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FuncDef {
    pub ty: DataType,
    pub name: String,
    pub params: Vec<FuncFParam>,
    pub body: Block,
    pub span: Span,
}

// CompUnit ::= [CompUnit] (Decl | FuncDef);
#[derive(Debug, Clone, PartialEq)]
pub enum GlobalItem {
    Decl(Decl),
    FuncDef(FuncDef),
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompUnit {
    pub items: Vec<GlobalItem>
}