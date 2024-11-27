package quantreg

import (
	"math"
	"testing"
)

// TestNLRQ tests non-linear quantile regression with a simple exponential model
func TestNLRQ(t *testing.T) {
	// Create synthetic data from exponential model: y = β₁ * exp(β₂ * x) + ε
	x := [][]float64{
		{0.0},
		{0.5},
		{1.0},
		{1.5},
		{2.0},
	}

	// True parameters
	beta1, beta2 := 1.0, 0.5
	y := make([]float64, len(x))
	for i, xi := range x {
		y[i] = beta1 * math.Exp(beta2*xi[0])
	}

	// Define exponential model
	model := NonLinearModel{
		// Model function: f(β, x) = β₁ * exp(β₂ * x)
		F: func(beta []float64, x []float64) float64 {
			return beta[0] * math.Exp(beta[1]*x[0])
		},
		// Gradient: [∂f/∂β₁, ∂f/∂β₂]
		// ∂f/∂β₁ = exp(β₂ * x)
		// ∂f/∂β₂ = β₁ * x * exp(β₂ * x)
		Gradient: func(beta []float64, x []float64) []float64 {
			exp := math.Exp(beta[1] * x[0])
			return []float64{
				exp,                    // ∂f/∂β₁
				beta[0] * x[0] * exp,   // ∂f/∂β₂
			}
		},
	}

	// Initial parameter values
	beta0 := []float64{0.5, 0.1}

	// Fit model
	fit, err := NLRQ(y, x, model, beta0, 0.5)
	if err != nil {
		t.Fatalf("Failed to fit model: %v", err)
	}

	// Check dimensions
	if fit.N != 5 {
		t.Errorf("Expected 5 observations, got %d", fit.N)
	}

	if fit.P != 2 {
		t.Errorf("Expected 2 parameters, got %d", fit.P)
	}

	// Check if estimated parameters are close to true values
	if math.Abs(fit.Coefficients[0]-beta1) > 0.1 {
		t.Errorf("β₁ estimate too far from true value: got %.2f, want %.2f",
			fit.Coefficients[0], beta1)
	}

	if math.Abs(fit.Coefficients[1]-beta2) > 0.1 {
		t.Errorf("β₂ estimate too far from true value: got %.2f, want %.2f",
			fit.Coefficients[1], beta2)
	}

	// Test predictions
	newX := [][]float64{
		{2.5},
		{3.0},
	}
	predictions, err := fit.Predict(newX)
	if err != nil {
		t.Fatalf("Failed to generate predictions: %v", err)
	}

	if len(predictions) != 2 {
		t.Errorf("Expected 2 predictions, got %d", len(predictions))
	}

	// Test error cases
	_, err = NLRQ(y, x, model, beta0, 1.5) // Invalid tau
	if err == nil {
		t.Error("Expected error for invalid tau")
	}

	_, err = NLRQ(y[:2], x, model, beta0, 0.5) // Mismatched dimensions
	if err == nil {
		t.Error("Expected error for mismatched dimensions")
	}
}

func TestNLRQFitSummary(t *testing.T) {
	// Simple exponential model
	model := NonLinearModel{
		F: func(beta []float64, x []float64) float64 {
			return beta[0] * math.Exp(beta[1]*x[0])
		},
		Gradient: func(beta []float64, x []float64) []float64 {
			exp := math.Exp(beta[1] * x[0])
			return []float64{exp, beta[0] * x[0] * exp}
		},
	}

	x := [][]float64{
		{0.0},
		{0.5},
		{1.0},
	}

	y := []float64{1.0, 1.65, 2.72}
	beta0 := []float64{1.0, 1.0}

	fit, err := NLRQ(y, x, model, beta0, 0.5)
	if err != nil {
		t.Fatalf("Failed to fit model: %v", err)
	}

	summary := fit.Summary()
	if summary == "" {
		t.Error("Expected non-empty summary string")
	}
}
