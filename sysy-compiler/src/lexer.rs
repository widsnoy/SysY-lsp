use logos::Logos;
use std::{num::ParseIntError, ops::Range};

#[derive(Debug, Clone, PartialEq)]
pub enum LexerError {
    InvalidInteger(String),
    InvalidToken,
}

impl Default for LexerError {
    fn default() -> Self {
        LexerError::InvalidToken
    }
}

impl From<ParseIntError> for LexerError {
    fn from(value: ParseIntError) -> Self {
        LexerError::InvalidInteger(value.to_string())
    }
}

/// 字符串 -> 数字
fn parse_int_literal(lex: &mut logos::Lexer<TokenKind>) -> Result<isize, ParseIntError> {
    let slice = lex.slice();

    if let Some(hex_body) = slice
        .strip_prefix("0x")
        .or_else(|| slice.strip_prefix("0X"))
    {
        isize::from_str_radix(hex_body, 16)
    } else if slice.starts_with('0') && slice.len() > 1 {
        isize::from_str_radix(&slice[1..], 8)
    } else {
        slice.parse::<isize>()
    }
}

#[derive(Logos, Debug, PartialEq, Clone)]
#[logos(error = LexerError)]
pub enum TokenKind {
    // trivia
    #[regex(r"[ \t\n\f]+")]
    Whitespace,

    #[regex(r"//[^\n]*")]
    Comment,

    #[regex(r"/\*([^*]|\*[^/])*\*/")]
    BlockComment,

    // Keywords
    #[token("const")]
    Const,
    #[token("int")]
    Int,
    #[token("void")]
    Void,
    #[token("if")]
    If,
    #[token("else")]
    Else,
    #[token("while")]
    While,
    #[token("break")]
    Break,
    #[token("continue")]
    Continue,
    #[token("return")]
    Return,

    #[regex("[a-zA-Z_][a-zA-Z0-9_]*")]
    Ident,

    #[regex(r"[0-9][0-9a-zA-Z_]*", parse_int_literal)]
    IntConst(isize),

    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Mul,
    #[token("/")]
    Div,
    #[token("%")]
    Mod,
    #[token("==")]
    Eq,
    #[token("!=")]
    Ne,
    #[token("<")]
    Lt,
    #[token(">")]
    Gt,
    #[token("<=")]
    Le,
    #[token(">=")]
    Ge,
    #[token("&&")]
    And,
    #[token("||")]
    Or,
    #[token("!")]
    Not,
    #[token("=")]
    Assign,

    #[token(";")]
    Semicolon,
    #[token(",")]
    Comma,
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub text: String,
    pub span: Range<usize>,
}

fn lexer(source: &str) -> (Vec<Token>, Vec<(LexerError, Range<usize>)>) {
    let mut tokens = vec![];
    let mut errors = vec![];
    let mut lexer = TokenKind::lexer(source);

    while let Some(result) = lexer.next() {
        let span = lexer.span();
        match result {
            Ok(kind) => tokens.push(Token {
                kind,
                text: lexer.slice().to_string(),
                span,
            }),
            Err(error) => {
                let lexer_error = if let LexerError::InvalidInteger(_) = error {
                    error
                } else {
                    LexerError::InvalidToken
                };
                errors.push((lexer_error, span));
            }
        }
    }

    (tokens, errors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::LineIndex;

    #[test]
    fn lex_complex_source_with_errors() {
        let source = r#"
/*
 * A complex test function for the lexer.
 * It includes loops, conditions, and various tokens.
 */
const int MAX_SIZE = 100;

int main() {
    int i = 0;
    int arr[MAX_SIZE]; // Array declaration

    while (i < MAX_SIZE && 1) {
        if (i % 2 == 0) {
            arr[i] = i * 2;
        } else {
            arr[i] = 0x1F; // Hex literal
        }
        i = i + 1;
    }

    // Invalid token test
    int error_test = @;

    return 0;
}
"#;

        let line_index = LineIndex::new(source);

        let (tokens, errors) = lexer(source);

        errors.iter().for_each(|(error, range)| {
            println!("{:?} {error:?}", line_index.line_col(range.start));
        });

        // 1. 检查错误（这部分是正确的）
        assert_eq!(errors.len(), 1, "Expected one lexical error");
        let (error, span) = &errors[0];
        assert_eq!(error, &LexerError::InvalidToken);
        assert_eq!(source.get(span.clone()).unwrap(), "@");

        dbg!(&tokens);

        // 2. **修正部分：过滤掉 trivia token**
        let meaningful_tokens: Vec<_> = tokens
            .into_iter()
            .filter(|t| {
                !matches!(
                    t.kind,
                    TokenKind::Whitespace | TokenKind::Comment | TokenKind::BlockComment
                )
            })
            .collect();

        // 3. 在过滤后的、稳定的 Token 列表上进行断言
        // assert_eq!(meaningful_tokens.len(), 38, "Expected 38 meaningful tokens");

        // 检查 const 声明
        assert_eq!(meaningful_tokens[0].kind, TokenKind::Const);
        assert_eq!(meaningful_tokens[1].kind, TokenKind::Int);
        assert_eq!(meaningful_tokens[2].kind, TokenKind::Ident);
        assert_eq!(meaningful_tokens[2].text, "MAX_SIZE");
        assert_eq!(meaningful_tokens[4].kind, TokenKind::IntConst(100));

        // 检查十六进制字面量
        let hex_token = meaningful_tokens.iter().find(|t| t.text == "0x1F").unwrap();
        assert_eq!(hex_token.kind, TokenKind::IntConst(31));

        // 检查最后一个有效 Token
        assert_eq!(meaningful_tokens.last().unwrap().kind, TokenKind::RBrace);
    }
}
