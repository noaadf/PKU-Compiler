use super::{Expr, Span};

// DataType ::= "int" | "void";
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DataType {
    Int,
    Void,
}

// Decl ::= ConstDecl | VarDecl;
#[derive(Debug, Clone, PartialEq)]
pub enum Decl {
    ConstDecl(ConstDecl),
    VarDecl(VarDecl),
}

// ConstDecl ::= "const" DataType ConstDef {"," ConstDef} ";";
#[derive(Debug, Clone, PartialEq)]
pub struct ConstDecl {
    pub ty: DataType,
    pub defs: Vec<ConstDef>,
    pub span: Span,
}

// ConstDef ::= IDENT {"[" ConstExp "]"} "=" ConstInitVal;
#[derive(Debug, Clone, PartialEq)]
pub struct ConstDef {
    pub name: String,
    pub dims: Vec<Expr>,  // 数组维度长度（ConstExp 列表）
    pub init: ConstInitVal,
    pub span: Span,
}

// ConstInitVal ::= ConstExp | "{" [ConstInitVal {"," ConstInitVal}] "}";
#[derive(Debug, Clone, PartialEq)]
pub enum ConstInitVal {
    Single(Expr),
    List(Vec<ConstInitVal>),
}

// VarDecl ::= DataType VarDef {"," VarDef} ";";
#[derive(Debug, Clone, PartialEq)]
pub struct VarDecl {
    pub ty: DataType,
    pub defs: Vec<VarDef>,
    pub span: Span,
}

// VarDef ::= IDENT {"[" ConstExp "]"} | IDENT {"[" ConstExp "]"} "=" InitVal;
#[derive(Debug, Clone, PartialEq)]
pub struct VarDef {
    pub name: String,
    pub dims: Vec<Expr>,  // 数组维度长度（ConstExp 列表）
    pub init: Option<InitVal>,
    pub span: Span,
}

// InitVal ::= Exp | "{" [InitVal {"," InitVal}] "}";
#[derive(Debug, Clone, PartialEq)]
pub enum InitVal {
    Single(Expr),
    List(Vec<InitVal>),
}

