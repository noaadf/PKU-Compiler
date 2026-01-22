use crate::back::insts::Reg;

#[derive(Clone)]
pub struct TargetRegInfo {
    pub allocatable: Vec<Reg>,
    pub caller_saved: Vec<Reg>,
    pub callee_saved: Vec<Reg>,
    pub scratch: Vec<Reg>,
}

impl TargetRegInfo {
    pub fn riscv() -> Self {
        let caller_saved = vec![
            Reg::T0, Reg::T1, Reg::T2, Reg::T3, Reg::T4, Reg::T5, Reg::T6,
        ];
        let callee_saved = vec![
            Reg::S0, Reg::S1, Reg::S2, Reg::S3, Reg::S4, Reg::S5, Reg::S6,
            Reg::S7, Reg::S8, Reg::S9, Reg::S10, Reg::S11,
        ];
        let scratch = vec![Reg::T4, Reg::T5, Reg::T6];
        let allocatable = vec![
            Reg::T0, Reg::T1, Reg::T2, Reg::T3,
            Reg::S0, Reg::S1, Reg::S2, Reg::S3, Reg::S4, Reg::S5,
            Reg::S6, Reg::S7, Reg::S8, Reg::S9, Reg::S10, Reg::S11,
        ];
        Self {
            allocatable,
            caller_saved,
            callee_saved,
            scratch,
        }
    }
}
