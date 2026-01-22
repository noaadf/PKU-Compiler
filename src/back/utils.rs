use crate::back::context::Context;
use crate::back::insts::{Instruction, Reg};

pub fn store_reg_with_offset(ctx: &mut Context, reg: Reg, offset: i32) {
    if offset >= -2048 && offset <= 2047 {
        ctx.program.push(Instruction::Sw(reg, offset, Reg::Sp));
    } else {
        let addr = ctx.alloc_reg();
        ctx.program.push(Instruction::Li(addr, offset));
        ctx.program.push(Instruction::Add(addr, Reg::Sp, addr));
        ctx.program.push(Instruction::Sw(reg, 0, addr));
    }
}

pub fn load_reg_with_offset(ctx: &mut Context, reg: Reg, offset: i32) {
    if offset >= -2048 && offset <= 2047 {
        ctx.program.push(Instruction::Lw(reg, offset, Reg::Sp));
    } else {
        let addr = ctx.alloc_reg();
        ctx.program.push(Instruction::Li(addr, offset));
        ctx.program.push(Instruction::Add(addr, Reg::Sp, addr));
        ctx.program.push(Instruction::Lw(reg, 0, addr));
    }
}
