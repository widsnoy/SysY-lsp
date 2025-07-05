use logos::Logos;
use std::num::ParseIntError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LexerError {
    InvalidInteger,
    InvalidToken,
}

impl Default for LexerError {
    fn default() -> Self {
        LexerError::InvalidToken
    }
}

impl From<ParseIntError> for LexerError {
    fn from(_: ParseIntError) -> Self {
        LexerError::InvalidInteger
    }
}

/// 字符串 -> 数字 (仅用于验证)
fn parse_int_literal(lex: &mut logos::Lexer<TokenKind>) -> Result<(), LexerError> {
    let slice = lex.slice();

    // Check for valid integer literal format first
    let value = if slice.starts_with("0x") || slice.starts_with("0X") {
        // Hexadecimal: 0x followed by hex digits
        let hex_body = &slice[2..];
        if hex_body.is_empty() || !hex_body.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(LexerError::InvalidInteger);
        }
        isize::from_str_radix(hex_body, 16)
    } else if slice.starts_with('0') && slice.len() > 1 {
        // Octal: 0 followed by octal digits (0-7)
        let octal_body = &slice[1..];
        if !octal_body.chars().all(|c| c >= '0' && c <= '7') {
            return Err(LexerError::InvalidInteger);
        }
        isize::from_str_radix(octal_body, 8)
    } else {
        // Decimal: should only contain digits
        if !slice.chars().all(|c| c.is_ascii_digit()) {
            return Err(LexerError::InvalidInteger);
        }
        slice.parse::<isize>()
    };

    value.map(|_| ()).map_err(|_| LexerError::InvalidInteger)
}

#[derive(Logos, Debug, PartialEq, Eq, Clone, Copy)]
#[logos(error = LexerError)]
pub enum TokenKind {
    // trivia
    #[regex(r"[ \t\n\f]+")]
    Whitespace,

    #[regex(r"//[^\n]*|/\*([^*]|\*[^/])*\*/")]
    Comment,

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
    IntConst,

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
    Leq,
    #[token(">=")]
    Geq,
    #[token("&&")]
    And,
    #[token("||")]
    Or,
    #[token("!")]
    Not,
    #[token("=")]
    Assign,
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token(",")]
    Comma,
    #[token(";")]
    Semicolon,

    Eof,
}

pub fn lex(
    code: &str,
) -> impl Iterator<Item = (Result<TokenKind, LexerError>, &str, std::ops::Range<usize>)> {
    TokenKind::lexer(code).spanned().map(|(tok, span)| {
        let text = &code[span.start..span.end];
        (tok, text, span)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use logos::Logos;

    #[test]
    fn lexer_test() {
        let source = r#"
            int main() {
                const int a = 10;
                int b = a + 20;
                // comment
                return b;
            }
        "#;

        let tokens: Vec<_> = TokenKind::lexer(source)
            .filter_map(|tok| tok.ok())
            .filter(|kind| !matches!(kind, TokenKind::Whitespace | TokenKind::Comment))
            .collect();

        let expected = vec![
            TokenKind::Int,
            TokenKind::Ident,
            TokenKind::LParen,
            TokenKind::RParen,
            TokenKind::LBrace,
            TokenKind::Const,
            TokenKind::Int,
            TokenKind::Ident,
            TokenKind::Assign,
            TokenKind::IntConst,
            TokenKind::Semicolon,
            TokenKind::Int,
            TokenKind::Ident,
            TokenKind::Assign,
            TokenKind::Ident,
            TokenKind::Plus,
            TokenKind::IntConst,
            TokenKind::Semicolon,
            TokenKind::Return,
            TokenKind::Ident,
            TokenKind::Semicolon,
            TokenKind::RBrace,
        ];

        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_integer_literals() {
        let test_cases = vec![
            // Valid decimal
            ("0", Ok(TokenKind::IntConst)),
            ("42", Ok(TokenKind::IntConst)),
            ("123", Ok(TokenKind::IntConst)),
            // Valid hexadecimal
            ("0x0", Ok(TokenKind::IntConst)),
            ("0x42", Ok(TokenKind::IntConst)),
            ("0XaB", Ok(TokenKind::IntConst)),
            ("0xff", Ok(TokenKind::IntConst)),
            // Valid octal
            ("01", Ok(TokenKind::IntConst)),
            ("07", Ok(TokenKind::IntConst)),
            ("0123", Ok(TokenKind::IntConst)),
            // Invalid cases that should be lexer errors
            ("0x", Err(LexerError::InvalidInteger)), // hex prefix without digits
            ("0xG", Err(LexerError::InvalidInteger)), // invalid hex digit
            ("08", Err(LexerError::InvalidInteger)), // invalid octal digit
            ("09", Err(LexerError::InvalidInteger)), // invalid octal digit
            ("123abc", Err(LexerError::InvalidInteger)), // mixed letters in decimal
            ("0xabc_def", Err(LexerError::InvalidInteger)), // underscore in hex
        ];

        for (input, expected) in test_cases {
            let tokens: Vec<_> = lex(input).collect();
            // Should have at least 1 token (the parsed token or error), maybe 2 with EOF
            assert!(!tokens.is_empty(), "No tokens for input: {}", input);

            let (result, text, _span) = &tokens[0];
            assert_eq!(text, &input);

            match expected {
                Ok(expected_kind) => {
                    assert_eq!(*result, Ok(expected_kind), "Failed for input: {}", input);
                }
                Err(expected_error) => {
                    assert_eq!(*result, Err(expected_error), "Failed for input: {}", input);
                }
            }
        }
    }

    #[test]
    fn test_token_positions() {
        let code = "int x = 42;";
        let tokens: Vec<_> = lex(code).collect();

        // Expected: int(0..3), whitespace(3..4), x(4..5), whitespace(5..6), =(6..7), whitespace(7..8), 42(8..10), ;(10..11)
        let expected_positions = vec![
            (0..3, "int"),
            (3..4, " "),
            (4..5, "x"),
            (5..6, " "),
            (6..7, "="),
            (7..8, " "),
            (8..10, "42"),
            (10..11, ";"),
        ];

        for (i, ((result, text, span), (expected_span, expected_text))) in
            tokens.iter().zip(expected_positions.iter()).enumerate()
        {
            assert!(result.is_ok(), "Token {} should be valid", i);
            assert_eq!(text, expected_text, "Token {} text mismatch", i);
            assert_eq!(span, expected_span, "Token {} span mismatch", i);
        }
    }
}
