use super::Span;

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    IntLiteral(i32, Span),
    // 变量引用：由 LVal 产生
    LVal(LVal),
    UnaryOp(UnaryOp, Box<Expr>, Span),
    BinaryOp(BinaryOp, Box<Expr>, Box<Expr>, Span),
    RelOp(RelOp, Box<Expr>, Box<Expr>, Span),
    EqOp(EqOp, Box<Expr>, Box<Expr>, Span),
    LAndOp(LAndOp, Box<Expr>, Box<Expr>, Span),
    LOrOp(LOrOp, Box<Expr>, Box<Expr>, Span),
    // 函数调用：函数名、参数列表、位置
    Call(String, Vec<Expr>, Span),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnaryOp {
    Plus,   // +
    Minus,  // -
    Not,    // !
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LiteralType {
    Int,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RelOp {
    Lt,   // <
    Gt,   // >
    Le,   // <=
    Ge,   // >=
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EqOp {
    Eq,   // ==
    Ne,   // !=
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LAndOp {
    And,  // &&
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LOrOp {
    Or,   // ||
}

// LVal ::= IDENT {"[" Exp "]"};
#[derive(Debug, Clone, PartialEq)]
pub struct LVal {
    pub name: String,
    pub indices: Vec<Expr>,  // 数组索引（Exp 列表）
    pub span: Span,
}