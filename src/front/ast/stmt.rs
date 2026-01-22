use super::{Expr, Span, Decl};

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    // "return" [Exp] ";"
    Return(Option<Expr>),
    // LVal "=" Exp ";"
    Assign(super::LVal, Expr),
    // Block
    Block(Block),
    // [Exp] ";"
    Expr(Option<Expr>),
    // "if" "(" Exp ")" Stmt ["else" Stmt]
    If {
        cond: Expr,
        then_stmt: Box<Stmt>,
        else_stmt: Option<Box<Stmt>>,
        span: Span,
    },
    // "while" "(" Exp ")" Stmt
    While {
        cond: Expr,
        body: Box<Stmt>,
        span: Span,
    },
    // "break" ";"
    Break(Span),
    // "continue" ";"
    Continue(Span),
}

// BlockItem ::= Decl | Stmt;
#[derive(Debug, Clone, PartialEq)]
pub enum BlockItem {
    Decl(Decl),
    Stmt(Stmt),
}

// Block ::= "{" {BlockItem} "}";
pub type Block = Vec<BlockItem>;