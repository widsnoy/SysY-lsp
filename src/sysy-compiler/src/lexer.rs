use logos::Logos;
use std::num::ParseIntError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LexerError {
    // InvalidInteger,
    InvalidToken,
}

impl Default for LexerError {
    fn default() -> Self {
        LexerError::InvalidToken
    }
}

// impl From<ParseIntError> for LexerError {
//     fn from(_: ParseIntError) -> Self {
//         LexerError::InvalidInteger
//     }
// }

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

    #[regex(r"[0-9][0-9a-zA-Z_]*")]
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
