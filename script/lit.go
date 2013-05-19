package script

import (
	"go/ast"
	"go/token"
	"strconv"
)

func (w *World) compileBasicLit(n *ast.BasicLit) expr {
	switch n.Kind {
	default:
		panic(err("not allowed:", n.Value, "(", typ(n), ")"))
	case token.INT, token.FLOAT:
		return floatLit(parseFloat(n.Value))
	}
}

type floatLit float64

func (l floatLit) Eval() []interface{} { return []interface{}{float64(l)} }
func (l floatLit) NumOut() int         { return 1 }

func parseFloat(str string) float64 {
	v, err := strconv.ParseFloat(str, 64)
	if err != nil {
		panic("internal error")
	}
	return v
}
