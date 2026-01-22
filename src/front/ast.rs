pub mod expr;
pub mod stmt;
pub mod func;
pub mod decl;

pub use expr::*;
pub use stmt::*;
pub use func::*;
pub use decl::*;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Span {
    pub begin: usize,
    pub end: usize,
}

impl Span {
    pub fn from_span(start: usize, end: usize) -> Self {
        Span { begin: start, end }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Ident {
    pub name: String,
}

impl Ident {
    pub fn new(name: String) -> Self {
        Self {
            name,
        }
    }
}


