package quantreg

import (
	"math"
	"testing"
)

func TestRQ(t *testing.T) {
	// Create a simple dataset
	x := [][]float64{
		{1, 0.5},
		{1, 1.0},
		{1, 1.5},
		{1, 2.0},
		{1, 2.5},
	}
	
	y := []float64{1.0, 2.0, 2.5, 3.0, 4.0}
	
	// Fit median regression (tau = 0.5)
	fit, err := RQ(y, x, 0.5)
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
	
	// Check coefficients are reasonable
	if math.Abs(fit.Coefficients[0]) > 10 || math.Abs(fit.Coefficients[1]) > 10 {
		t.Errorf("Unreasonable coefficient estimates: %v", fit.Coefficients)
	}
	
	// Test predictions
	predictions, err := fit.Predict(x)
	if err != nil {
		t.Fatalf("Failed to generate predictions: %v", err)
	}
	
	if len(predictions) != 5 {
		t.Errorf("Expected 5 predictions, got %d", len(predictions))
	}
	
	// Test error cases
	_, err = RQ(y, x, 1.5) // Invalid tau
	if err == nil {
		t.Error("Expected error for invalid tau")
	}
	
	_, err = RQ(y[:2], x, 0.5) // Mismatched dimensions
	if err == nil {
		t.Error("Expected error for mismatched dimensions")
	}
}

func TestRQFitSummary(t *testing.T) {
	x := [][]float64{
		{1, 0.5},
		{1, 1.0},
		{1, 1.5},
	}
	
	y := []float64{1.0, 2.0, 2.5}
	
	fit, err := RQ(y, x, 0.5)
	if err != nil {
		t.Fatalf("Failed to fit model: %v", err)
	}
	
	summary := fit.Summary()
	if summary == "" {
		t.Error("Expected non-empty summary string")
	}
}
