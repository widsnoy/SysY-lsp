pub struct LineIndex {
    line_starts: Vec<usize>,
}

impl LineIndex {
    pub fn new(source: &str) -> Self {
        let mut line_starts = vec![0];
        line_starts.extend(source.match_indices('\n').map(|(i, _)| i + 1));
        Self { line_starts }
    }

    /// 根据文件字符偏移获取行列号
    pub fn line_col(&self, offset: usize) -> (usize, usize) {
        let line = self.line_starts.partition_point(|&x| x <= offset) - 1;
        let column = offset - self.line_starts[line];
        (line, column)
    }
}
