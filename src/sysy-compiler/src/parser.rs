use crate::lexer::{LexerError, TokenKind, lex};
use rowan::{GreenNode, GreenNodeBuilder};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SysYLanguage;

impl rowan::Language for SysYLanguage {
    type Kind = SyntaxKind;
    fn kind_from_raw(raw: rowan::SyntaxKind) -> Self::Kind {
        assert!(raw.0 <= SyntaxKind::Root as u16);
        unsafe { std::mem::transmute::<u16, SyntaxKind>(raw.0) }
    }
    fn kind_to_raw(kind: Self::Kind) -> rowan::SyntaxKind {
        kind.into()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(non_camel_case_types)]
#[repr(u16)]
pub enum SyntaxKind {
    Whitespace,
    Comment,
    Error, // Represents a lexical or syntax error
    Eof,   // End of file token

    // Tokens
    Ident,
    IntConst,
    Const,
    Int,
    Void,
    If,
    Else,
    While,
    Break,
    Continue,
    Return,
    Assign,
    Plus,
    Minus,
    Mul,
    Div,
    Mod,
    Eq,
    Neq,
    Lt,
    Gt,
    Leq,
    Geq,
    Not,
    And,
    Or,
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Semicolon,

    // Rules
    CompUnit,
    Decl,
    ConstDecl,
    BType,
    ConstDef,
    ConstInitVal,
    VarDecl,
    VarDef,
    InitVal,
    FuncDef,
    FuncType,
    FuncFParams,
    FuncFParam,
    Block,
    BlockItem,
    Stmt,
    Exp,
    Cond,
    LVal,
    PrimaryExp,
    Number,
    UnaryExp,
    UnaryOp,
    FuncRParams,
    MulExp,
    AddExp,
    RelExp,
    EqExp,
    LAndExp,
    LOrExp,
    ConstExp,

    // Must be last
    Root,
}

impl From<TokenKind> for SyntaxKind {
    fn from(token: TokenKind) -> Self {
        match token {
            TokenKind::Whitespace => Self::Whitespace,
            TokenKind::Comment => Self::Comment,
            TokenKind::Ident => Self::Ident,
            TokenKind::IntConst => Self::IntConst,
            TokenKind::Const => Self::Const,
            TokenKind::Int => Self::Int,
            TokenKind::Void => Self::Void,
            TokenKind::If => Self::If,
            TokenKind::Else => Self::Else,
            TokenKind::While => Self::While,
            TokenKind::Break => Self::Break,
            TokenKind::Continue => Self::Continue,
            TokenKind::Return => Self::Return,
            TokenKind::Assign => Self::Assign,
            TokenKind::Plus => Self::Plus,
            TokenKind::Minus => Self::Minus,
            TokenKind::Mul => Self::Mul,
            TokenKind::Div => Self::Div,
            TokenKind::Mod => Self::Mod,
            TokenKind::Eq => Self::Eq,
            TokenKind::Ne => Self::Neq,
            TokenKind::Lt => Self::Lt,
            TokenKind::Gt => Self::Gt,
            TokenKind::Leq => Self::Leq,
            TokenKind::Geq => Self::Geq,
            TokenKind::Not => Self::Not,
            TokenKind::And => Self::And,
            TokenKind::Or => Self::Or,
            TokenKind::LParen => Self::LParen,
            TokenKind::RParen => Self::RParen,
            TokenKind::LBrace => Self::LBrace,
            TokenKind::RBrace => Self::RBrace,
            TokenKind::LBracket => Self::LBracket,
            TokenKind::RBracket => Self::RBracket,
            TokenKind::Comma => Self::Comma,
            TokenKind::Semicolon => Self::Semicolon,
            TokenKind::Eof => Self::Eof,
        }
    }
}

impl From<SyntaxKind> for rowan::SyntaxKind {
    fn from(kind: SyntaxKind) -> Self {
        Self(kind as u16)
    }
}

pub struct Parse {
    pub green_node: GreenNode,
    pub errors: Vec<String>,
}

/// Parses the source code and returns a `Parse` result, which includes
/// the green tree and a list of errors.
pub fn parse(source: &str) -> Parse {
    let mut tokens: Vec<_> = lex(source)
        .map(|(result, text, _span)| (result, text.to_string()))
        .collect();
    tokens.push((Ok(TokenKind::Eof), String::new()));

    let mut parser = Parser::new(&tokens);
    parser.parse_comp_unit();

    let (builder, errors) = parser.finish();
    Parse {
        green_node: builder.finish(),
        errors,
    }
}

/// The main parser structure. It consumes a token stream and builds a green tree.
struct Parser<'a> {
    /// The stream of tokens, reversed for efficient popping.
    tokens: Vec<(Result<TokenKind, LexerError>, &'a str)>,
    /// The builder for the Rowan green tree.
    builder: GreenNodeBuilder<'static>,
    /// A list of errors encountered during parsing.
    errors: Vec<String>,
}

impl<'a> Parser<'a> {
    /// Creates a new parser from a token stream.
    /// The token stream is reversed to allow for efficient `pop` operations.
    fn new(tokens: &'a [(Result<TokenKind, LexerError>, String)]) -> Self {
        let tokens_rev = tokens
            .iter()
            .map(|(kind, s)| (*kind, s.as_str()))
            .rev()
            .collect();
        Self {
            tokens: tokens_rev,
            builder: GreenNodeBuilder::new(),
            errors: Vec::new(),
        }
    }

    /// Finishes parsing and returns the builder and errors.
    fn finish(self) -> (GreenNodeBuilder<'static>, Vec<String>) {
        (self.builder, self.errors)
    }

    /// Peeks at the next non-trivia token kind without consuming it.
    /// Automatically skips trivia tokens (whitespace, comments).
    /// Handles lexical errors by treating them as `SyntaxKind::Error`.
    fn peek(&self) -> Option<SyntaxKind> {
        for (result, _) in self.tokens.iter().rev() {
            match result {
                Ok(kind) => {
                    let syntax_kind: SyntaxKind = (*kind).into();
                    if !matches!(syntax_kind, SyntaxKind::Whitespace | SyntaxKind::Comment) {
                        return Some(syntax_kind);
                    }
                }
                Err(_) => return Some(SyntaxKind::Error),
            }
        }
        None
    }

    /// Consumes the next token and adds it to the tree.
    /// Automatically consumes any trivia tokens first.
    fn bump(&mut self) {
        // First consume all trivia tokens
        self.eat_trivia();

        // Then consume the next non-trivia token
        if let Some((result, text)) = self.tokens.pop() {
            let kind = match result {
                Ok(kind) => kind.into(),
                Err(e) => {
                    self.errors.push(format!("LexerError: {:?}", e));
                    SyntaxKind::Error
                }
            };
            self.builder.token(kind.into(), text);
        }
    }

    /// Consumes trivia tokens (whitespace, comments) and adds them to the tree.
    /// This function directly examines the token stream without using peek().
    fn eat_trivia(&mut self) {
        while let Some((result, _text)) = self.tokens.last() {
            let is_trivia = match result {
                Ok(kind) => {
                    let syntax_kind: SyntaxKind = (*kind).into();
                    matches!(syntax_kind, SyntaxKind::Whitespace | SyntaxKind::Comment)
                }
                Err(_) => false, // Don't treat errors as trivia
            };

            if is_trivia {
                // Remove the token from the stream and add it to the tree
                let (result, text) = self.tokens.pop().unwrap();
                let kind = match result {
                    Ok(kind) => kind.into(),
                    Err(e) => {
                        self.errors.push(format!("LexerError: {:?}", e));
                        SyntaxKind::Error
                    }
                };
                self.builder.token(kind.into(), text);
            } else {
                break;
            }
        }
    }

    /// Parses a `CompUnit`, the root of a SysY source file.
    /// CompUnit → [ CompUnit ] ( Decl | FuncDef )
    /// 等价于: CompUnit → { Decl | FuncDef }+
    fn parse_comp_unit(&mut self) {
        self.builder.start_node(SyntaxKind::CompUnit.into());

        // 必须至少有一个 Decl 或 FuncDef
        let mut has_items = false;

        loop {
            self.eat_trivia();
            match self.peek() {
                Some(SyntaxKind::Eof) => {
                    self.bump(); // Consume Eof
                    break;
                }
                Some(SyntaxKind::Const) => {
                    self.parse_decl();
                    has_items = true;
                }
                Some(SyntaxKind::Int) => {
                    // 需要前瞻来区分 VarDecl 和 FuncDef
                    if self.is_func_def() {
                        self.parse_func_def();
                    } else {
                        self.parse_decl();
                    }
                    has_items = true;
                }
                Some(SyntaxKind::Void) => {
                    self.parse_func_def();
                    has_items = true;
                }
                Some(SyntaxKind::Error) => {
                    // Handle lexical error token
                    self.bump();
                }
                None => break, // End of token stream
                _ => {
                    // Unexpected token at the top level
                    self.errors
                        .push(format!("Unexpected token: {:?}", self.peek()));
                    // Create an error node to consume the token and advance
                    self.builder.start_node(SyntaxKind::Error.into());
                    self.bump();
                    self.builder.finish_node();
                }
            }
        }

        if !has_items {
            self.errors.push(
                "CompUnit must contain at least one declaration or function definition".to_string(),
            );
        }

        self.builder.finish_node(); // Finish CompUnit
    }

    /// 解析声明
    /// Decl → ConstDecl | VarDecl
    fn parse_decl(&mut self) {
        self.builder.start_node(SyntaxKind::Decl.into());

        match self.peek() {
            Some(SyntaxKind::Const) => {
                self.parse_const_decl();
            }
            Some(SyntaxKind::Int) => {
                // 在这个上下文中，int 只能是 VarDecl
                // 因为 CompUnit 层面已经区分了 Decl 和 FuncDef
                self.parse_var_decl();
            }
            _ => {
                self.errors
                    .push("Expected 'const' or 'int' for declaration".to_string());
            }
        }

        self.builder.finish_node();
    }

    /// 解析常量声明
    /// ConstDecl → 'const' BType ConstDef { ',' ConstDef } ';'
    fn parse_const_decl(&mut self) {
        self.builder.start_node(SyntaxKind::ConstDecl.into());

        // 'const'
        if self.peek() == Some(SyntaxKind::Const) {
            self.bump();
        } else {
            self.errors.push("Expected 'const'".to_string());
        }

        // BType
        self.parse_btype();

        // ConstDef
        self.parse_const_def();

        // { ',' ConstDef }
        while self.peek() == Some(SyntaxKind::Comma) {
            self.bump(); // consume ','
            self.parse_const_def();
        }

        // ';'
        if self.peek() == Some(SyntaxKind::Semicolon) {
            self.bump();
        } else {
            self.errors
                .push("Expected ';' after constant declaration".to_string());
        }

        self.builder.finish_node();
    }

    /// 解析基本类型
    /// BType → 'int' | 'float'  (这里只实现 'int')
    fn parse_btype(&mut self) {
        self.builder.start_node(SyntaxKind::BType.into());

        match self.peek() {
            Some(SyntaxKind::Int) => {
                self.bump();
            }
            _ => {
                self.errors.push("Expected 'int' type".to_string());
            }
        }

        self.builder.finish_node();
    }

    /// 解析常数定义
    /// ConstDef → Ident { '[' ConstExp ']' } '=' ConstInitVal
    fn parse_const_def(&mut self) {
        self.builder.start_node(SyntaxKind::ConstDef.into());

        // Ident
        if self.peek() == Some(SyntaxKind::Ident) {
            self.bump();
        } else {
            self.errors
                .push("Expected identifier in constant definition".to_string());
        }

        // { '[' ConstExp ']' } - 数组维度，暂时跳过
        while self.peek() == Some(SyntaxKind::LBracket) {
            self.bump(); // '['
            self.parse_const_exp();
            if self.peek() == Some(SyntaxKind::RBracket) {
                self.bump(); // ']'
            } else {
                self.errors.push("Expected ']'".to_string());
            }
        }

        // '='
        if self.peek() == Some(SyntaxKind::Assign) {
            self.bump();
        } else {
            self.errors
                .push("Expected '=' in constant definition".to_string());
        }

        // ConstInitVal
        self.parse_const_init_val();

        self.builder.finish_node();
    }

    /// 解析常量初值
    /// ConstInitVal → ConstExp | '{' [ ConstInitVal { ',' ConstInitVal } ] '}'
    fn parse_const_init_val(&mut self) {
        self.builder.start_node(SyntaxKind::ConstInitVal.into());

        if self.peek() == Some(SyntaxKind::LBrace) {
            // 数组初始化：'{' [ ConstInitVal { ',' ConstInitVal } ] '}'
            self.bump(); // '{'

            // [ ConstInitVal { ',' ConstInitVal } ] - 可选的初始化列表
            if self.peek() != Some(SyntaxKind::RBrace) {
                // 第一个 ConstInitVal
                self.parse_const_init_val();

                // { ',' ConstInitVal }
                while self.peek() == Some(SyntaxKind::Comma) {
                    self.bump(); // ','
                    self.parse_const_init_val();
                }
            }

            if self.peek() == Some(SyntaxKind::RBrace) {
                self.bump(); // '}'
            } else {
                self.errors
                    .push("Expected '}' after array initialization".to_string());
            }
        } else {
            // 单个常量表达式
            self.parse_const_exp();
        }

        self.builder.finish_node();
    }

    /// 解析变量声明
    /// VarDecl → BType VarDef { ',' VarDef } ';'
    fn parse_var_decl(&mut self) {
        self.builder.start_node(SyntaxKind::VarDecl.into());

        // BType
        self.parse_btype();

        // VarDef
        self.parse_var_def();

        // { ',' VarDef }
        while self.peek() == Some(SyntaxKind::Comma) {
            self.bump(); // ','
            self.parse_var_def();
        }

        // ';'
        if self.peek() == Some(SyntaxKind::Semicolon) {
            self.bump();
        } else {
            self.errors
                .push("Expected ';' after variable declaration".to_string());
        }

        self.builder.finish_node();
    }

    /// 解析变量定义
    /// VarDef → Ident { '[' ConstExp ']' } '=' InitVal
    fn parse_var_def(&mut self) {
        self.builder.start_node(SyntaxKind::VarDef.into());

        // Ident
        if self.peek() == Some(SyntaxKind::Ident) {
            self.bump();
        } else {
            self.errors
                .push("Expected identifier in variable definition".to_string());
        }

        // { '[' ConstExp ']' }
        while self.peek() == Some(SyntaxKind::LBracket) {
            self.bump(); // '['
            self.parse_const_exp();
            if self.peek() == Some(SyntaxKind::RBracket) {
                self.bump(); // ']'
            } else {
                self.errors.push("Expected ']'".to_string());
            }
        }

        // [ '=' InitVal ]
        if self.peek() == Some(SyntaxKind::Assign) {
            self.bump(); // '='
            self.parse_init_val();
        }

        self.builder.finish_node();
    }

    /// 解析变量初值
    /// InitVal → Exp | '{' [ InitVal { ',' InitVal } ] '}'
    fn parse_init_val(&mut self) {
        self.builder.start_node(SyntaxKind::InitVal.into());

        if self.peek() == Some(SyntaxKind::LBrace) {
            // 数组初始化：'{' [ InitVal { ',' InitVal } ] '}'
            self.bump(); // '{'

            // [ InitVal { ',' InitVal } ] - 可选的初始化列表
            if self.peek() != Some(SyntaxKind::RBrace) {
                // 第一个 InitVal
                self.parse_init_val();

                // { ',' InitVal }
                while self.peek() == Some(SyntaxKind::Comma) {
                    self.bump(); // ','
                    self.parse_init_val();
                }
            }

            if self.peek() == Some(SyntaxKind::RBrace) {
                self.bump(); // '}'
            } else {
                self.errors
                    .push("Expected '}' after array initialization".to_string());
            }
        } else {
            // 表达式
            self.parse_exp();
        }

        self.builder.finish_node();
    }

    /// 简化的前瞻检查：判断是否是函数定义
    /// 通过查找是否有 '(' 来判断
    fn is_func_def(&self) -> bool {
        // 简化实现：向前查找几个 token 看是否有 '('
        for (i, (result, _)) in self.tokens.iter().rev().enumerate() {
            if i > 5 {
                break;
            } // 只查看前几个 token
            if let Ok(token) = result {
                match token {
                    TokenKind::LParen => return true,
                    TokenKind::Semicolon | TokenKind::Assign => return false,
                    _ => continue,
                }
            }
        }
        false
    }

    /// 暂时的占位实现
    fn parse_const_exp(&mut self) {
        self.builder.start_node(SyntaxKind::ConstExp.into());
        // TODO: 实现常量表达式解析
        if self.peek() == Some(SyntaxKind::IntConst) {
            self.bump();
        } else {
            self.errors.push("Expected constant expression".to_string());
        }
        self.builder.finish_node();
    }

    /// 暂时的占位实现
    fn parse_exp(&mut self) {
        self.builder.start_node(SyntaxKind::Exp.into());
        // TODO: 实现表达式解析
        while self.peek() != Some(SyntaxKind::Semicolon)
            && self.peek() != Some(SyntaxKind::Comma)
            && self.peek() != Some(SyntaxKind::RBrace)
            && self.peek() != Some(SyntaxKind::Eof)
        {
            self.bump();
        }
        self.builder.finish_node();
    }

    /// 解析函数定义
    /// FuncDef → FuncType Ident '(' [FuncFParams] ')' Block
    fn parse_func_def(&mut self) {
        self.builder.start_node(SyntaxKind::FuncDef.into());

        // FuncType
        self.parse_func_type();

        // Ident
        if self.peek() == Some(SyntaxKind::Ident) {
            self.bump();
        } else {
            self.errors.push("Expected function name".to_string());
        }

        // '('
        if self.peek() == Some(SyntaxKind::LParen) {
            self.bump();
        } else {
            self.errors
                .push("Expected '(' after function name".to_string());
        }

        // [FuncFParams]
        if self.peek() != Some(SyntaxKind::RParen) {
            self.parse_func_fparams();
        }

        // ')'
        if self.peek() == Some(SyntaxKind::RParen) {
            self.bump();
        } else {
            self.errors
                .push("Expected ')' after function parameters".to_string());
        }

        // Block
        self.parse_block();

        self.builder.finish_node();
    }

    /// 解析函数类型
    /// FuncType → 'void' | 'int' | 'float'
    fn parse_func_type(&mut self) {
        self.builder.start_node(SyntaxKind::FuncType.into());

        match self.peek() {
            Some(SyntaxKind::Void) | Some(SyntaxKind::Int) => {
                self.bump();
            }
            _ => {
                self.errors
                    .push("Expected function return type ('void' or 'int')".to_string());
            }
        }

        self.builder.finish_node();
    }

    /// 解析函数形参表
    /// FuncFParams → FuncFParam { ',' FuncFParam }
    fn parse_func_fparams(&mut self) {
        self.builder.start_node(SyntaxKind::FuncFParams.into());

        // FuncFParam
        self.parse_func_fparam();

        // { ',' FuncFParam }
        while self.peek() == Some(SyntaxKind::Comma) {
            self.bump(); // ','
            self.parse_func_fparam();
        }

        self.builder.finish_node();
    }

    /// 解析函数形参
    /// FuncFParam → BType Ident ['[' ']' { '[' Exp ']' }]
    fn parse_func_fparam(&mut self) {
        self.builder.start_node(SyntaxKind::FuncFParam.into());

        // BType
        self.parse_btype();

        // Ident
        if self.peek() == Some(SyntaxKind::Ident) {
            self.bump();
        } else {
            self.errors.push("Expected parameter name".to_string());
        }

        // ['[' ']' { '[' Exp ']' }] - 数组参数，暂时简化
        if self.peek() == Some(SyntaxKind::LBracket) {
            self.bump(); // '['
            if self.peek() == Some(SyntaxKind::RBracket) {
                self.bump(); // ']'
                // { '[' Exp ']' }
                while self.peek() == Some(SyntaxKind::LBracket) {
                    self.bump(); // '['
                    self.parse_exp();
                    if self.peek() == Some(SyntaxKind::RBracket) {
                        self.bump(); // ']'
                    } else {
                        self.errors.push("Expected ']'".to_string());
                    }
                }
            } else {
                self.errors.push("Expected ']' after '['".to_string());
            }
        }

        self.builder.finish_node();
    }

    /// 解析语句块
    /// Block → '{' { BlockItem } '}'
    fn parse_block(&mut self) {
        self.builder.start_node(SyntaxKind::Block.into());

        // '{'
        if self.peek() == Some(SyntaxKind::LBrace) {
            self.bump();
        } else {
            self.errors.push("Expected '{'".to_string());
        }

        // { BlockItem }
        while self.peek() != Some(SyntaxKind::RBrace) && self.peek() != Some(SyntaxKind::Eof) {
            self.eat_trivia();
            match self.peek() {
                Some(SyntaxKind::Const) | Some(SyntaxKind::Int) => {
                    if self.is_func_def() {
                        self.errors
                            .push("Function definitions not allowed inside blocks".to_string());
                        break;
                    } else {
                        self.parse_decl();
                    }
                }
                Some(SyntaxKind::RBrace) => break,
                Some(_) => {
                    self.parse_stmt();
                }
                None => break,
            }
        }

        // '}'
        if self.peek() == Some(SyntaxKind::RBrace) {
            self.bump();
        } else {
            self.errors.push("Expected '}'".to_string());
        }

        self.builder.finish_node();
    }

    /// 解析语句 (简化版本)
    /// Stmt → LVal '=' Exp ';' | [Exp] ';' | Block | ...
    fn parse_stmt(&mut self) {
        self.builder.start_node(SyntaxKind::Stmt.into());

        match self.peek() {
            Some(SyntaxKind::LBrace) => {
                // Block
                self.parse_block();
            }
            Some(SyntaxKind::If) => {
                // TODO: if statement
                self.bump();
                self.errors
                    .push("If statements not yet implemented".to_string());
            }
            Some(SyntaxKind::While) => {
                // TODO: while statement
                self.bump();
                self.errors
                    .push("While statements not yet implemented".to_string());
            }
            Some(SyntaxKind::Return) => {
                // return [Exp] ';'
                self.bump(); // 'return'
                if self.peek() != Some(SyntaxKind::Semicolon) {
                    self.parse_exp();
                }
                if self.peek() == Some(SyntaxKind::Semicolon) {
                    self.bump();
                } else {
                    self.errors
                        .push("Expected ';' after return statement".to_string());
                }
            }
            Some(SyntaxKind::Break) | Some(SyntaxKind::Continue) => {
                // 'break' ';' | 'continue' ';'
                self.bump();
                if self.peek() == Some(SyntaxKind::Semicolon) {
                    self.bump();
                } else {
                    self.errors
                        .push("Expected ';' after break/continue".to_string());
                }
            }
            _ => {
                // [Exp] ';' or LVal '=' Exp ';'
                if self.peek() != Some(SyntaxKind::Semicolon) {
                    self.parse_exp();
                }
                if self.peek() == Some(SyntaxKind::Semicolon) {
                    self.bump();
                } else {
                    self.errors.push("Expected ';' after statement".to_string());
                }
            }
        }

        self.builder.finish_node();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex;

    /// 辅助函数：解析代码并返回错误信息
    fn parse_and_get_errors(source: &str) -> Vec<String> {
        let result = parse(source);
        result.errors
    }

    /// 辅助函数：检查解析是否成功（没有错误）
    fn parse_success(source: &str) -> bool {
        let errors = parse_and_get_errors(source);
        if !errors.is_empty() {
            println!("Parse errors for '{}': {:?}", source, errors);
            // 也打印词法分析的结果
            let tokens: Vec<_> = lex(source).collect();
            println!("Tokens: {:?}", tokens);
        }
        errors.is_empty()
    }

    #[test]
    fn test_simple_const_decl() {
        // 简单常量声明
        assert!(parse_success("const int a = 5;"));
        assert!(parse_success("const int x = 42;"));

        // 多个常量声明
        assert!(parse_success("const int a = 1, b = 2, c = 3;"));
    }

    #[test]
    fn test_simple_var_decl() {
        // 简单变量声明
        assert!(parse_success("int a;"));
        assert!(parse_success("int x = 10;"));

        // 多个变量声明
        assert!(parse_success("int a, b, c;"));
        assert!(parse_success("int x = 1, y = 2;"));
    }

    #[test]
    fn test_array_declarations() {
        // 数组声明（常量）
        assert!(parse_success("const int arr[5] = {1, 2, 3, 4, 5};"));
        assert!(parse_success(
            "const int matrix[2][3] = {{1, 2, 3}, {4, 5, 6}};"
        ));

        // 数组声明（变量）
        assert!(parse_success("int arr[10];"));
        assert!(parse_success("int data[5] = {1, 2, 3, 4, 5};"));
    }

    #[test]
    fn test_empty_array_init() {
        // 空数组初始化
        assert!(parse_success("int arr[] = {};"));
        assert!(parse_success("const int empty[0] = {};"));
    }

    #[test]
    fn test_simple_function_def() {
        // 简单函数定义
        assert!(parse_success("void main() {}"));
        assert!(parse_success("int add() { return 42; }"));
    }

    #[test]
    fn test_function_with_params() {
        // 带参数的函数
        assert!(parse_success("int add(int a, int b) { return a; }"));
        assert!(parse_success("void print(int x) {}"));

        // 数组参数
        assert!(parse_success("int sum(int arr[]) { return 0; }"));
        assert!(parse_success("void process(int matrix[][10]) {}"));
    }

    #[test]
    fn test_function_with_statements() {
        // 包含各种语句的函数
        let code = r#"
            int main() {
                int a = 5;
                return a;
            }
        "#;
        assert!(parse_success(code));

        let code2 = r#"
            void test() {
                break;
                continue;
                return;
            }
        "#;
        assert!(parse_success(code2));
    }

    #[test]
    fn test_nested_blocks() {
        // 嵌套块
        let code = r#"
            void test() {
                {
                    int x = 1;
                    {
                        int y = 2;
                    }
                }
            }
        "#;
        assert!(parse_success(code));
    }

    #[test]
    fn test_mixed_declarations_and_functions() {
        // 混合声明和函数定义
        let code = r#"
            const int CONSTANT = 100;
            int global_var;
            
            int add(int a, int b) {
                return a;
            }
            
            void main() {
                int local = 5;
            }
        "#;
        assert!(parse_success(code));
    }

    #[test]
    fn test_complex_array_init() {
        // 复杂数组初始化
        let code = r#"
            const int matrix[2][2] = {{1, 2}, {3, 4}};
        "#;
        assert!(parse_success(code));

        let code2 = r#"
            int tensor[2][2][2] = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
        "#;
        assert!(parse_success(code2));
    }

    #[test]
    fn test_error_cases() {
        // 语法错误的情况
        let errors = parse_and_get_errors("const int;"); // 缺少标识符
        assert!(!errors.is_empty());

        let errors2 = parse_and_get_errors("int x = ;"); // 缺少初始值
        assert!(!errors2.is_empty());

        let errors3 = parse_and_get_errors("void func( {}"); // 缺少右括号
        assert!(!errors3.is_empty());
    }

    #[test]
    fn test_empty_program() {
        // 空程序应该报错
        let errors = parse_and_get_errors("");
        assert!(!errors.is_empty());
        assert!(
            errors
                .iter()
                .any(|e| e.contains("must contain at least one"))
        );
    }

    #[test]
    fn test_comments_and_whitespace() {
        // 带注释和空白的代码
        let code = r#"
            // This is a comment
            const int x = 5; /* another comment */
            
            int main() {
                // return something
                return 0;
            }
        "#;
        assert!(parse_success(code));
    }

    #[test]
    fn test_function_vs_variable_distinction() {
        // 测试函数和变量声明的区分
        // assert!(parse_success("int func();")); // 这会被当作变量声明（简化版） 沟槽的 ai，哪里有这种语法
        assert!(parse_success("int func(int x) { return x; }")); // 这是函数定义
    }

    #[test]
    fn test_const_expressions() {
        // 常量表达式（目前只支持简单数字）
        assert!(parse_success("const int a = 42;"));
        assert!(parse_success("const int arr[5] = {1, 2, 3, 4, 5};"));

        // 数组大小中的常量表达式
        assert!(parse_success("int arr[10];"));
        assert!(parse_success("const int matrix[3][4] = {};"));
    }

    #[test]
    fn test_various_statements() {
        // 各种语句类型
        let code = r#"
            void test() {
                int x;           // 声明语句
                x = 5;           // 表达式语句
                return x;        // return 语句
                break;           // break 语句  
                continue;        // continue 语句
                ;                // 空语句
                {                // 块语句
                    int y = 10;
                }
            }
        "#;
        assert!(parse_success(code));
    }

    #[test]
    fn test_multiple_compilation_units() {
        // 多个顶层项目
        let code = r#"
            const int CONST1 = 1;
            int var1;
            const int CONST2 = 2;
            
            int func1() { return 1; }
            
            int var2 = 5;
            
            void func2() {}
        "#;
        assert!(parse_success(code));
    }
}
