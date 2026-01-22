#[derive(Debug, Clone)]
pub struct SourceMap {
    line_starts: Vec<usize>,
}

impl SourceMap {
    pub fn new(input: &str) -> Self {
        let mut line_starts = vec![0];
        for (idx, byte) in input.bytes().enumerate() {
            if byte == b'\n' {
                line_starts.push(idx + 1);
            }
        }
        Self { line_starts }
    }

    pub fn line_col(&self, offset: usize) -> (usize, usize) {
        let line_idx = match self.line_starts.binary_search(&offset) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        };
        let line_start = self.line_starts.get(line_idx).copied().unwrap_or(0);
        let line = line_idx + 1;
        let col = offset.saturating_sub(line_start) + 1;
        (line, col)
    }

    pub fn format_location(&self, offset: usize) -> String {
        let (line, col) = self.line_col(offset);
        format!("line {}, column {}", line, col)
    }
}
