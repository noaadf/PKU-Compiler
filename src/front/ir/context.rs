use koopa::ir::dfg::DataFlowGraph;
use koopa::ir::layout::Layout;
use koopa::ir::*;
use koopa::ir::builder_traits::*;
use crate::front::ast::*;
use std::collections::HashMap;
use crate::front::ir::stmt_ir::*;
use crate::front::ir::expr_ir::*;
use crate::CompilerError;
use crate::utils::SourceMap;
pub struct IrContext {
    pub program: Program,
    pub current_func: Option<Function>,
    pub current_bb: Option<BasicBlock>,
    pub scopes: Vec<(i32, HashMap<String, Value>)>,
    pub source_map: Option<SourceMap>,
    /// 全局作用域：存储全局变量和常量
    pub global_scope: HashMap<String, Value>,
    /// 全局常量集合：记录哪些全局符号是常量（Integer 类型）
    pub global_constants: std::collections::HashSet<String>,
    /// 记录"当前语义块"结束后应跳转到的基本块（用于嵌套 if）
    pub next_bb: Option<BasicBlock>,
    /// 循环栈：记录每一层循环的 (continue_target_bb, break_target_bb)
    pub loop_stack: Vec<(BasicBlock, BasicBlock)>,
    /// 函数符号表：函数名 -> Function handle
    pub func_table: HashMap<String, Function>,
    /// 当前函数的数组形参维度数（首维省略时也计入）
    pub array_param_dims: HashMap<String, usize>,
    /// 当前函数入口基本块（用于放置一次性 alloc）
    pub entry_bb: Option<BasicBlock>,
}

impl IrContext{
    pub fn new(source_map: Option<SourceMap>) -> Self{
        Self {
            program: Program::new(),
            current_func: None,
            current_bb: None,
            scopes: Vec::new(),
            source_map,
            global_scope: HashMap::new(),
            global_constants: std::collections::HashSet::new(),
            next_bb: None,
            loop_stack: Vec::new(),
            func_table: HashMap::new(),
            array_param_dims: HashMap::new(),
            entry_bb: None,
        }
    }

    pub fn format_location(&self, offset: usize) -> Option<String> {
        self.source_map.as_ref().map(|sm| sm.format_location(offset))
    }

    pub fn error_at_span(&self, message: impl Into<String>, span: Span) -> CompilerError {
        let mut msg = message.into();
        if let Some(loc) = self.format_location(span.begin) {
            msg = format!("{} at {}", msg, loc);
        }
        CompilerError::IRGenerationError(msg)
    }

    pub fn get_current_bb(&mut self) -> Result<BasicBlock, CompilerError> {
        self.current_bb.ok_or_else(|| CompilerError::IRGenerationError("No current basic block".to_string()))
    }

    pub fn dfg_mut(&mut self) -> Result<&mut DataFlowGraph, CompilerError> {
        let f = self.current_func.ok_or_else(|| CompilerError::IRGenerationError("No current function".to_string()))?;
        Ok(self.program.func_mut(f).dfg_mut())
    }

    pub fn layout(&mut self) -> Result<&mut Layout, CompilerError> {
        let f = self.current_func.ok_or_else(|| CompilerError::IRGenerationError("No current function".to_string()))?;
        Ok(self.program.func_mut(f).layout_mut())
    }

    /// Push a new scope onto the stack
    pub fn push_scope(&mut self) {
        let next_level = self.scopes.last().map(|s| s.0 + 1).unwrap_or(0);
        self.scopes.push((next_level, HashMap::new()));
    }

    /// Pop the current scope from the stack
    pub fn pop_scope(&mut self) -> Result<(), CompilerError> {
        self.scopes.pop().ok_or_else(|| CompilerError::IRGenerationError("Cannot pop scope: no scope on stack".to_string()))?;
        Ok(())
    }

    /// Look up a variable in the scope stack (from innermost to outermost)
    pub fn lookup_var(&self, name: &str) -> Option<Value> {
        for scope in self.scopes.iter().rev() {
            if let Some(&val) = scope.1.get(name) {
                return Some(val);
            }
        }
        self.global_scope.get(name).copied()
    }

    /// Insert a variable into the current scope
    /// Returns an error if the variable is already declared in the current scope
    pub fn insert_var(&mut self, name: String, value: Value) -> Result<(), CompilerError> {
        let current_scope = self.scopes.last_mut()
            .ok_or_else(|| CompilerError::IRGenerationError("Cannot insert variable: no scope on stack".to_string()))?;
        if current_scope.1.contains_key(&name) {
            return Err(CompilerError::IRGenerationError(format!("Variable `{}` already declared in this scope", name)));
        }
        current_scope.1.insert(name, value);
        Ok(())
    }

    pub fn insert_global_var(&mut self, name: String, value: Value) -> Result<(), CompilerError> {
        if self.global_scope.contains_key(&name) {
            return Err(CompilerError::IRGenerationError(format!("Global variable `{}` already declared", name)));
        }
        self.global_scope.insert(name, value);
        Ok(())
    }

    /// Create a new basic block and add it to the function layout.
    /// 不指定名字，让 Koopa 自己命名，避免兼容性问题。
    pub fn new_bb(&mut self, _name: Option<String>) -> Result<BasicBlock, CompilerError> {
        let bb = self.dfg_mut()?.new_bb().basic_block(None);
        self.layout()?.bbs_mut().push_key_back(bb)
            .map_err(|_| CompilerError::IRGenerationError("Failed to add basic block".to_string()))?;
        Ok(bb)
    }

    /// Set the current basic block
    pub fn set_current_bb(&mut self, bb: BasicBlock) {
        self.current_bb = Some(bb);
    }

    pub fn has_terminator(&mut self, bb: BasicBlock) -> Result<bool, CompilerError> {
        let last_inst = self
            .layout()?
            .bb_mut(bb)
            .insts()
            .keys()
            .last()
            .copied();
        if let Some(inst) = last_inst {
            let value_data = self.dfg_mut()?.value(inst);
            Ok(matches!(
                value_data.kind(),
                ValueKind::Return(_) | ValueKind::Jump(_) | ValueKind::Branch(_)
            ))
        } else {
            Ok(false)
        }
    }

    /// 在函数入口块中插入 alloc（避免在循环中反复分配）
    pub fn alloc_in_entry(&mut self, ty: Type) -> Result<Value, CompilerError> {
        let entry_bb = self.entry_bb.ok_or_else(|| {
            CompilerError::IRGenerationError("No entry basic block for current function".to_string())
        })?;
        let alloc_inst = {
            let dfg = self.dfg_mut()?;
            dfg.new_value().alloc(ty)
        };
        // 放在入口块最前，避免落在 terminator 之后导致 IR 非法
        self.layout()?
            .bb_mut(entry_bb)
            .insts_mut()
            .push_key_front(alloc_inst)
            .map_err(|_| CompilerError::IRGenerationError("Failed to add alloc instruction to entry".to_string()))?;
        Ok(alloc_inst)
    }
    
    pub fn push_inst(&mut self, inst: Value) -> Result<(), CompilerError> {
        let bb = self.current_bb.ok_or_else(|| CompilerError::IRGenerationError("No current basic block".to_string()))?;
        self.layout()?
            .bb_mut(bb)
            .insts_mut()
            .push_key_back(inst)
            .map_err(|_| CompilerError::IRGenerationError("Failed to insert instruction".to_string()))
    }
}