pub mod stmt_ir;
pub mod expr_ir;
pub mod context;

use koopa::ir::dfg::DataFlowGraph;
use koopa::ir::layout::Layout;
use koopa::ir::*;
use koopa::ir::builder_traits::*;
use crate::front::ast::*;
use std::collections::HashMap;
use crate::front::ir::stmt_ir::*;
use crate::front::ir::expr_ir::*;
use crate::CompilerError;
pub use crate::front::ir::context::IrContext;

pub trait GenerateIR {
    type Output;
    fn generate_ir(&self, ctx: &mut IrContext) -> Result<Self::Output, CompilerError>;
}

impl GenerateIR for CompUnit {
    type Output = ();

    fn generate_ir(&self, ctx: &mut IrContext) -> Result<Self::Output, CompilerError> {
        // 预先声明所有 SysY 库函数
        declare_library_functions(ctx)?;

        for item in &self.items {
            match item {
                GlobalItem::Decl(decl) => {
                    decl.generate_ir(ctx)?;
                }
                GlobalItem::FuncDef(_) => {}
            }
        }

        // 预先声明所有用户函数（此时全局常量已可用于参数维度）
        for item in &self.items {
            if let GlobalItem::FuncDef(func) = item {
                if ctx.func_table.contains_key(&func.name) {
                    return Err(CompilerError::IRGenerationError(format!(
                        "Function `{}` already declared",
                        func.name
                    )));
                }
                let mut param_tys = Vec::new();
                for param in &func.params {
                    let (param_ty, _) = build_param_type(param, ctx)?;
                    param_tys.push(param_ty);
                }
                let ret_ty = match func.ty {
                    DataType::Int => Type::get_i32(),
                    DataType::Void => Type::get_unit(),
                };
                let func_name = format!("@{}", func.name);
                let func_data = FunctionData::new(func_name, param_tys, ret_ty);
                let f = ctx.program.new_func(func_data);
                ctx.func_table.insert(func.name.clone(), f);
            }
        }

        for item in &self.items {
            match item {
                GlobalItem::FuncDef(func) => {
                    func.generate_ir(ctx)?;
                }
                GlobalItem::Decl(_) => {}
            }
        }
        Ok(())
    }
}

/// 声明所有 SysY 库函数
fn declare_library_functions(ctx: &mut IrContext) -> Result<(), CompilerError> {
    // getint(): i32
    let getint_func = create_lib_func_decl(
        ctx,
        "@getint".to_string(),
        vec![],
        Type::get_i32(),
    )?;
    ctx.func_table.insert("getint".to_string(), getint_func);
    
    // getch(): i32
    let getch_func = create_lib_func_decl(
        ctx,
        "@getch".to_string(),
        vec![],
        Type::get_i32(),
    )?;
    ctx.func_table.insert("getch".to_string(), getch_func);
    
    // getarray(*i32): i32
    let getarray_func = create_lib_func_decl(
        ctx,
        "@getarray".to_string(),
        vec![Type::get_pointer(Type::get_i32())],
        Type::get_i32(),
    )?;
    ctx.func_table.insert("getarray".to_string(), getarray_func);
    
    // putint(i32)
    let putint_func = create_lib_func_decl(
        ctx,
        "@putint".to_string(),
        vec![Type::get_i32()],
        Type::get_unit(),
    )?;
    ctx.func_table.insert("putint".to_string(), putint_func);
    
    // putch(i32)
    let putch_func = create_lib_func_decl(
        ctx,
        "@putch".to_string(),
        vec![Type::get_i32()],
        Type::get_unit(),
    )?;
    ctx.func_table.insert("putch".to_string(), putch_func);
    
    // putarray(i32, *i32)
    let putarray_func = create_lib_func_decl(
        ctx,
        "@putarray".to_string(),
        vec![Type::get_i32(), Type::get_pointer(Type::get_i32())],
        Type::get_unit(),
    )?;
    ctx.func_table.insert("putarray".to_string(), putarray_func);
    
    // starttime()
    let starttime_func = create_lib_func_decl(
        ctx,
        "@starttime".to_string(),
        vec![],
        Type::get_unit(),
    )?;
    ctx.func_table.insert("starttime".to_string(), starttime_func);
    
    // stoptime()
    let stoptime_func = create_lib_func_decl(
        ctx,
        "@stoptime".to_string(),
        vec![],
        Type::get_unit(),
    )?;
    ctx.func_table.insert("stoptime".to_string(), stoptime_func);
    
    Ok(())
}

/// 创建库函数声明（不添加基本块）
fn create_lib_func_decl(
    ctx: &mut IrContext,
    name: String,
    param_tys: Vec<Type>,
    ret_ty: Type,
) -> Result<Function, CompilerError> {
    let func_data = FunctionData::new(name, param_tys, ret_ty);
    let f = ctx.program.new_func(func_data);
    // 注意：不添加基本块，这样 entry_bb() 会返回 None，表示这是函数声明
    Ok(f)
}

fn product_dims(dims: &[usize]) -> usize {
    dims.iter().product()
}

fn build_array_type_from_dims(dims: &[usize]) -> Type {
    let mut ty = Type::get_i32();
    for dim in dims.iter().rev() {
        ty = Type::get_array(ty, *dim);
    }
    ty
}

fn build_param_type(
    param: &FuncFParam,
    ctx: &IrContext,
) -> Result<(Type, Option<Vec<usize>>), CompilerError> {
    if !param.is_array {
        return Ok((Type::get_i32(), None));
    }
    let mut dims = Vec::with_capacity(param.dims.len());
    for dim_expr in &param.dims {
        let dim = evaluate_const_expr(dim_expr, ctx)?;
        dims.push(dim as usize);
    }
    let base_ty = if dims.is_empty() {
        Type::get_i32()
    } else {
        build_array_type_from_dims(&dims)
    };
    Ok((Type::get_pointer(base_ty), Some(dims)))
}

fn pick_aligned_sub_dims<'a>(filled: usize, dims: &'a [usize]) -> Option<&'a [usize]> {
    // 从当前维度的下一级开始，选择“最大且对齐”的子数组维度
    // 这样多维数组初始化时，嵌套列表优先对齐到更高层的子数组
    for k in 1..dims.len() {
        let sub_size = product_dims(&dims[k..]);
        if filled % sub_size == 0 {
            return Some(&dims[k..]);
        }
    }
    None
}

fn eval_const_init_list(
    elems: &[ConstInitVal],
    dims: &[usize],
    ctx: &IrContext,
) -> Result<Vec<i32>, CompilerError> {
    if dims.is_empty() {
        return Err(CompilerError::IRGenerationError(
            "Braces around scalar initializer are not allowed".to_string(),
        ));
    }
    let total = product_dims(dims);
    let mut vals = Vec::with_capacity(total);
    let mut filled = 0;
    for elem in elems {
        if filled >= total {
            break;
        }
        match elem {
            ConstInitVal::Single(expr) => {
                vals.push(evaluate_const_expr(expr, ctx)?);
                filled += 1;
            }
            ConstInitVal::List(list) => {
                let sub_dims = pick_aligned_sub_dims(filled, dims)
                    .ok_or_else(|| CompilerError::IRGenerationError(
                        "Initializer list is not aligned with array dimensions".to_string(),
                    ))?;
                let sub_size = product_dims(sub_dims);
                let mut sub_vals = eval_const_init_list(list, sub_dims, ctx)?;
                vals.append(&mut sub_vals);
                filled += sub_size;
            }
        }
    }
    while vals.len() < total {
        vals.push(0);
    }
    Ok(vals)
}

fn eval_init_list(
    elems: &[InitVal],
    dims: &[usize],
) -> Result<Vec<Option<Expr>>, CompilerError> {
    if dims.is_empty() {
        return Err(CompilerError::IRGenerationError(
            "Braces around scalar initializer are not allowed".to_string(),
        ));
    }
    let total = product_dims(dims);
    let mut vals = Vec::with_capacity(total);
    let mut filled = 0;
    for elem in elems {
        if filled >= total {
            break;
        }
        match elem {
            InitVal::Single(expr) => {
                vals.push(Some(expr.clone()));
                filled += 1;
            }
            InitVal::List(list) => {
                let sub_dims = pick_aligned_sub_dims(filled, dims)
                    .ok_or_else(|| CompilerError::IRGenerationError(
                        "Initializer list is not aligned with array dimensions".to_string(),
                    ))?;
                let sub_size = product_dims(sub_dims);
                let mut sub_vals = eval_init_list(list, sub_dims)?;
                vals.append(&mut sub_vals);
                filled += sub_size;
            }
        }
    }
    while vals.len() < total {
        vals.push(None);
    }
    Ok(vals)
}

fn linear_to_indices(mut idx: usize, dims: &[usize]) -> Vec<usize> {
    let mut indices = Vec::with_capacity(dims.len());
    for (i, _dim) in dims.iter().enumerate() {
        let stride = product_dims(&dims[i + 1..]);
        let div = if stride == 0 { 1 } else { stride };
        indices.push(idx / div);
        idx %= div;
    }
    indices
}

fn build_global_aggregate_from_flat(
    flat_vals: &[i32],
    dims: &[usize],
    ctx: &mut IrContext,
) -> Result<Value, CompilerError> {
    if dims.is_empty() {
        return Err(CompilerError::IRGenerationError(
            "Invalid aggregate dimensions".to_string(),
        ));
    }
    if dims.len() == 1 {
        let vals: Vec<Value> = flat_vals
            .iter()
            .take(dims[0])
            .map(|v| ctx.program.new_value().integer(*v))
            .collect();
        return Ok(ctx.program.new_value().aggregate(vals));
    }
    let sub_size = product_dims(&dims[1..]);
    let mut elems = Vec::with_capacity(dims[0]);
    for i in 0..dims[0] {
        let start = i * sub_size;
        let end = start + sub_size;
        let sub_val = build_global_aggregate_from_flat(&flat_vals[start..end], &dims[1..], ctx)?;
        elems.push(sub_val);
    }
    Ok(ctx.program.new_value().aggregate(elems))
}

fn create_local_array_alloc(
    name: &str,
    array_type: Type,
    ctx: &mut IrContext,
) -> Result<Value, CompilerError> {
    let alloc_inst = ctx.dfg_mut()?.new_value().alloc(array_type);
    let scope_level = ctx.scopes.last().map(|s| s.0).unwrap_or(0);
    ctx.dfg_mut()?
        .set_value_name(alloc_inst, Some(format!("%{}_{}", name, scope_level)));
    ctx.push_inst(alloc_inst)?;
    Ok(alloc_inst)
}

fn store_local_array_i32(
    alloc_inst: Value,
    vals: &[i32],
    dims: &[usize],
    ctx: &mut IrContext,
) -> Result<(), CompilerError> {
    for (idx, val) in vals.iter().enumerate() {
        let indices = linear_to_indices(idx, dims);
        let mut ptr = alloc_inst;
        for index in indices {
            let idx_val = ctx.dfg_mut()?.new_value().integer(index as i32);
            let next_ptr = ctx.dfg_mut()?.new_value().get_elem_ptr(ptr, idx_val);
            ctx.push_inst(next_ptr)?;
            ptr = next_ptr;
        }
        let int_val = ctx.dfg_mut()?.new_value().integer(*val);
        let store_inst = ctx.dfg_mut()?.new_value().store(int_val, ptr);
        ctx.push_inst(store_inst)?;
    }
    Ok(())
}

fn store_local_array_exprs(
    alloc_inst: Value,
    elems: &[Option<Expr>],
    dims: &[usize],
    ctx: &mut IrContext,
) -> Result<(), CompilerError> {
    for (idx, elem) in elems.iter().enumerate() {
        let indices = linear_to_indices(idx, dims);
        let mut ptr = alloc_inst;
        for index in indices {
            let idx_val = ctx.dfg_mut()?.new_value().integer(index as i32);
            let next_ptr = ctx.dfg_mut()?.new_value().get_elem_ptr(ptr, idx_val);
            ctx.push_inst(next_ptr)?;
            ptr = next_ptr;
        }
        let val = if let Some(expr) = elem {
            expr.generate_ir(ctx)?
        } else {
            ctx.dfg_mut()?.new_value().integer(0)
        };
        let store_inst = ctx.dfg_mut()?.new_value().store(val, ptr);
        ctx.push_inst(store_inst)?;
    }
    Ok(())
}

impl GenerateIR for FuncDef {
    type Output = ();

    fn generate_ir(&self, ctx: &mut IrContext) -> Result<Self::Output, CompilerError> {
        let func_name = format!("@{}", self.name);
        
        // 构建参数列表（只包含类型）
        let mut param_tys = Vec::new();
        let mut param_dims = Vec::new();
        for param in &self.params {
            let (param_ty, dims) = build_param_type(param, ctx)?;
            param_tys.push(param_ty);
            param_dims.push(dims);
        }
        
        // 根据返回类型设置函数返回类型
        let ret_ty = match self.ty {
            DataType::Int => Type::get_i32(),
            DataType::Void => Type::get_unit(),
        };
        
        let f = if let Some(&f) = ctx.func_table.get(&self.name) {
            f
        } else {
            let func_data = FunctionData::new(
                func_name.clone(),
                param_tys.clone(),
                ret_ty,
            );
            let f = ctx.program.new_func(func_data);
            ctx.func_table.insert(self.name.clone(), f);
            f
        };
        ctx.current_func = Some(f);
        ctx.array_param_dims.clear();
        
        // 将函数添加到符号表
        ctx.func_table.insert(self.name.clone(), f);

        // 让 Koopa 自己分配基本块名（保持兼容性）
        let bb = ctx.dfg_mut()?.new_bb().basic_block(None);
        ctx.layout()?.bbs_mut().push_key_back(bb)
            .map_err(|_| CompilerError::IRGenerationError("Failed to add basic block".to_string()))?;
        ctx.current_bb = Some(bb);
        ctx.entry_bb = Some(bb);

        // Push initial scope for function body
        ctx.push_scope();

        // 处理函数参数：为每个参数分配内存并存储
        let param_values: Vec<Value> = {
            let func_data_ref = ctx.program.func_mut(f);
            func_data_ref.params().iter().copied().collect()
        };
        
        for (idx, param) in self.params.iter().enumerate() {
            // 参数值在函数参数列表中，使用参数索引访问
            let param_value = param_values[idx];

            // 为参数分配内存
            let param_alloc_ty = if param.is_array {
                param_tys[idx].clone()
            } else {
                Type::get_i32()
            };
            let param_alloc = ctx.dfg_mut()?.new_value().alloc(param_alloc_ty);
            ctx.layout()?.bb_mut(bb).insts_mut().push_key_back(param_alloc)
                .map_err(|_| CompilerError::IRGenerationError("Failed to add param alloc instruction".to_string()))?;
            
            // 将参数值存储到分配的内存中
            let store_inst = ctx.dfg_mut()?.new_value().store(param_value, param_alloc);
            ctx.layout()?.bb_mut(bb).insts_mut().push_key_back(store_inst)
                .map_err(|_| CompilerError::IRGenerationError("Failed to add param store instruction".to_string()))?;
            
            // 将参数地址添加到符号表
            ctx.insert_var(param.name.clone(), param_alloc)?;
            if param.is_array {
                let total_dims = 1 + param_dims[idx].as_ref().map(|d| d.len()).unwrap_or(0);
                ctx.array_param_dims.insert(param.name.clone(), total_dims);
            }
        }

        for item in &self.body {
            // 如果当前基本块已经有终结指令（return/jump/branch），跳过后续不可达代码
            // 注意：需要在每次迭代时重新获取 current_bb，因为前面的语句（如 while）可能已经改变了它
            let current_bb = ctx.get_current_bb()?;
            if ctx.has_terminator(current_bb)? { break; }
            
            match item {
                BlockItem::Decl(decl) => {
                    decl.generate_ir(ctx)?;
                }
                BlockItem::Stmt(stmt) => {
                    stmt.generate_ir(ctx)?;
                    let current_bb = ctx.get_current_bb()?;
                    if ctx.has_terminator(current_bb)? { break; }
                }
            }
        }
        let current_bb = ctx.get_current_bb()?;
        // 确保函数有返回语句：如果当前基本块没有终结指令，根据返回类型添加默认的 ret
        if !ctx.has_terminator(current_bb)? {
            match self.ty {
                DataType::Int => {
                    let zero = ctx.dfg_mut()?.new_value().integer(0);
                    let ret_inst = ctx.dfg_mut()?.new_value().ret(Some(zero));
                    ctx.push_inst(ret_inst)?;
                }
                DataType::Void => {
                    let ret_inst = ctx.dfg_mut()?.new_value().ret(None);
                    ctx.push_inst(ret_inst)?;
                }
            }
        }

        // Pop function body scope
        ctx.pop_scope()?;
        ctx.entry_bb = None;
        Ok(())
    }
}

impl GenerateIR for Stmt {
    type Output = ();

    fn generate_ir(&self, ctx: &mut IrContext) -> Result<Self::Output, CompilerError> {
        match self {
            Stmt::Return(expr) => generate_return_stmt_ir(expr, ctx),
            Stmt::Assign(lval, expr) => generate_assign_stmt_ir(lval, expr, ctx),
            Stmt::Block(block) => generate_block_stmt_ir(block, ctx),
            Stmt::Expr(expr) => generate_expr_stmt_ir(expr, ctx),
            Stmt::If { cond, then_stmt, else_stmt, .. } =>
                generate_if_stmt_ir(cond, then_stmt, else_stmt.as_deref(), ctx),
            Stmt::While { cond, body, .. } =>
                generate_while_stmt_ir(cond, body, ctx),
            Stmt::Break(span) => generate_break_stmt_ir(span, ctx),
            Stmt::Continue(span) => generate_continue_stmt_ir(span, ctx),
        }
    }
}

impl GenerateIR for Expr {
    type Output = Value;

    fn generate_ir(&self, ctx: &mut IrContext) -> Result<Self::Output, CompilerError> {
        match self {
            Expr::IntLiteral(n, _) => {
                if ctx.current_func.is_none() {
                    Ok(ctx.program.new_value().integer(*n))
                } else {
                    Ok(ctx.dfg_mut()?.new_value().integer(*n))
                }
            },
            Expr::LVal(lval) => generate_lval_ir(lval, ctx),
            Expr::UnaryOp(op, expr, _) => generate_unary_op_ir(op, expr, ctx),
            Expr::BinaryOp(op, lhs, rhs, _) => generate_binary_op_ir(op, lhs, rhs, ctx),
            Expr::RelOp(op, lhs, rhs, _) => generate_rel_op_ir(op, lhs, rhs, ctx),
            Expr::EqOp(op, lhs, rhs, _) => generate_eq_op_ir(op, lhs, rhs, ctx),
            Expr::LAndOp(_, lhs, rhs, _) => generate_land_op_ir(lhs, rhs, ctx),
            Expr::LOrOp(_, lhs, rhs, _) => generate_lor_op_ir(lhs, rhs, ctx),
            Expr::Call(name, args, span) => generate_call_ir(name, args, *span, ctx),
        }
    }
}

impl GenerateIR for Decl {
    type Output = ();

    fn generate_ir(&self, ctx: &mut IrContext) -> Result<Self::Output, CompilerError> {
        match self {
            Decl::ConstDecl(decl) => decl.generate_ir(ctx)?,
            Decl::VarDecl(decl) => decl.generate_ir(ctx)?,
        }
        Ok(())
    }
}

impl GenerateIR for ConstDecl {
    type Output = ();

    fn generate_ir(&self, ctx: &mut IrContext) -> Result<Self::Output, CompilerError> {
        let is_global = ctx.current_func.is_none();
        for def in &self.defs {
            // 判断是否为数组
            if !def.dims.is_empty() {
                let mut dims = Vec::with_capacity(def.dims.len());
                for dim_expr in &def.dims {
                    let dim = evaluate_const_expr(dim_expr, ctx)?;
                    dims.push(dim as usize);
                }
                let array_type = build_array_type_from_dims(&dims);

                match &def.init {
                    crate::front::ast::ConstInitVal::Single(_) => {
                        return Err(CompilerError::IRGenerationError(
                            "Const array must be initialized with a list".to_string()
                        ));
                    }
                    crate::front::ast::ConstInitVal::List(elems) => {
                        let vals = eval_const_init_list(elems, &dims, ctx)?;
                        if is_global {
                            let aggregate = build_global_aggregate_from_flat(&vals, &dims, ctx)?;
                            let global_alloc = ctx.program.new_value().global_alloc(aggregate);
                            let global_name = format!("@g_{}", def.name);
                            ctx.program.set_value_name(global_alloc, Some(global_name.clone()));
                            ctx.insert_global_var(def.name.clone(), global_alloc)?;
                        } else {
                            let alloc_inst = create_local_array_alloc(&def.name, array_type, ctx)?;
                            store_local_array_i32(alloc_inst, &vals, &dims, ctx)?;
                            ctx.insert_var(def.name.clone(), alloc_inst)?;
                        }
                    }
                }
            } else {
                // 标量常量：编译时求值，直接存储计算后的值
                match &def.init {
                    crate::front::ast::ConstInitVal::Single(expr) => {
                        let init_val = if is_global {
                            // 全局常量：使用 evaluate_const_expr 求值，创建整数常量
                            let val = evaluate_const_expr(expr, ctx)?;
                            ctx.program.new_value().integer(val)
                        } else {
                            // 局部常量：必须是常量表达式，直接求值为整数
                            let val = evaluate_const_expr(expr, ctx)?;
                            ctx.dfg_mut()?.new_value().integer(val)
                        };
                        
                        if is_global {
                            ctx.insert_global_var(def.name.clone(), init_val)?;
                            // 记录这是全局常量
                            ctx.global_constants.insert(def.name.clone());
                        } else {
                            ctx.insert_var(def.name.clone(), init_val)?;
                        }
                    }
                    crate::front::ast::ConstInitVal::List(_) => {
                        return Err(CompilerError::IRGenerationError(
                            "Scalar const cannot be initialized with a list".to_string()
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

impl GenerateIR for VarDecl {
    type Output = ();

    fn generate_ir(&self, ctx: &mut IrContext) -> Result<Self::Output, CompilerError> {
        // 检查是否在全局作用域（没有 current_func）
        let is_global = ctx.current_func.is_none();
        
        for def in &self.defs {
            // 判断是否为数组
            if !def.dims.is_empty() {
                let mut dims = Vec::with_capacity(def.dims.len());
                for dim_expr in &def.dims {
                    let dim = evaluate_const_expr(dim_expr, ctx)?;
                    dims.push(dim as usize);
                }
                let array_type = build_array_type_from_dims(&dims);

                if is_global {
                    let global_name = format!("@g_{}", def.name);
                    let init_val = match &def.init {
                        Some(crate::front::ast::InitVal::Single(_)) => {
                            return Err(CompilerError::IRGenerationError(
                                "Global array must be initialized with a list".to_string()
                            ));
                        }
                        Some(crate::front::ast::InitVal::List(elems)) => {
                            let flattened = eval_init_list(elems, &dims)?;
                            let mut vals = Vec::with_capacity(flattened.len());
                            for elem in flattened {
                                if let Some(expr) = elem {
                                    vals.push(evaluate_const_expr(&expr, ctx)?);
                                } else {
                                    vals.push(0);
                                }
                            }
                            build_global_aggregate_from_flat(&vals, &dims, ctx)?
                        }
                        None => ctx.program.new_value().zero_init(array_type),
                    };
                    let global_alloc = ctx.program.new_value().global_alloc(init_val);
                    ctx.program.set_value_name(global_alloc, Some(global_name.clone()));
                    ctx.insert_global_var(def.name.clone(), global_alloc)?;
                } else {
                    let alloc_inst = create_local_array_alloc(&def.name, array_type, ctx)?;
                    if let Some(crate::front::ast::InitVal::List(elems)) = &def.init {
                        let flattened = eval_init_list(elems, &dims)?;
                        store_local_array_exprs(alloc_inst, &flattened, &dims, ctx)?;
                    } else if let Some(crate::front::ast::InitVal::Single(_)) = &def.init {
                        return Err(CompilerError::IRGenerationError(
                            "Local array must be initialized with a list".to_string()
                        ));
                    }
                    ctx.insert_var(def.name.clone(), alloc_inst)?;
                }
            } else {
                // 标量变量：原有逻辑
                if is_global {
                    // 全局变量：生成 global alloc
                    let global_name = format!("@g_{}", def.name);
                    
                    let init_val = if let Some(init) = &def.init {
                        match init {
                            crate::front::ast::InitVal::Single(expr) => {
                                // 有初始值：计算常量表达式
                                let val = evaluate_const_expr(expr, ctx)?;
                                ctx.program.new_value().integer(val)
                            }
                            crate::front::ast::InitVal::List(_) => {
                                // 标量变量不支持列表初始化
                                return Err(CompilerError::IRGenerationError(
                                    "Scalar variable cannot be initialized with a list".to_string()
                                ));
                            }
                        }
                    } else {
                        // 未初始化的全局变量：使用 zeroinit
                        ctx.program.new_value().zero_init(Type::get_i32())
                    };
                    
                    // 创建全局 alloc，传入初始值
                    let global_alloc = ctx.program.new_value().global_alloc(init_val);
                    ctx.program.set_value_name(global_alloc, Some(global_name.clone()));

                    ctx.insert_global_var(def.name.clone(), global_alloc)?;
                } else {
                    // 局部变量：分配内存（原有逻辑）
                    let alloc_inst = ctx.dfg_mut()?.new_value().alloc(Type::get_i32());
                    let scope_level = ctx.scopes.last().map(|s| s.0).unwrap_or(0);
                    ctx.dfg_mut()?.set_value_name(alloc_inst, Some(format!("%{}_{}", def.name, scope_level)));
                    ctx.push_inst(alloc_inst)?;
                    
                    // 如果有初始值，生成 store 指令
                    if let Some(init) = &def.init {
                        match init {
                            crate::front::ast::InitVal::Single(expr) => {
                                let init_val = expr.generate_ir(ctx)?;
                                let store_inst = ctx.dfg_mut()?.new_value().store(init_val, alloc_inst);
                                ctx.push_inst(store_inst)?;
                            }
                            crate::front::ast::InitVal::List(_) => {
                                // 标量变量不支持列表初始化
                                return Err(CompilerError::IRGenerationError(
                                    "Scalar variable cannot be initialized with a list".to_string()
                                ));
                            }
                        }
                    }
                    
                    // 将 alloc 的地址存入当前作用域
                    ctx.insert_var(def.name.clone(), alloc_inst)?;
                }
            }
        }
        Ok(())
    }
}