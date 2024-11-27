package quantreg

import (
	"math"
	"testing"
)

func TestRQProcess(t *testing.T) {
	// Create synthetic data
	x := [][]float64{
		{1, 0.5},
		{1, 1.0},
		{1, 1.5},
		{1, 2.0},
		{1, 2.5},
	}
	y := []float64{1.0, 2.0, 2.5, 3.0, 4.0}

	// Test multiple quantile levels
	taus := []float64{0.25, 0.5, 0.75}
	fits, err := RQProcess(y, x, taus)
	if err != nil {
		t.Fatalf("Failed to fit models: %v", err)
	}

	// Check dimensions
	if fits.N != 5 {
		t.Errorf("Expected 5 observations, got %d", fits.N)
	}

	if fits.P != 2 {
		t.Errorf("Expected 2 parameters, got %d", fits.P)
	}

	if len(fits.Taus) != 3 {
		t.Errorf("Expected 3 quantile levels, got %d", len(fits.Taus))
	}

	// Test predictions
	newX := [][]float64{
		{1, 3.0},
		{1, 3.5},
	}
	predictions, err := fits.Predict(newX)
	if err != nil {
		t.Fatalf("Failed to generate predictions: %v", err)
	}

	if len(predictions) != 3 {
		t.Errorf("Expected predictions for 3 quantiles, got %d", len(predictions))
	}

	// Test diagnostics
	diag := fits.ComputeDiagnostics()
	if diag == nil {
		t.Fatal("Failed to compute diagnostics")
	}

	// Check residual stats
	for _, tau := range taus {
		stats, ok := diag.ResidualStats[tau]
		if !ok {
			t.Errorf("Missing residual stats for tau=%f", tau)
			continue
		}

		if math.IsNaN(stats.Mean) || math.IsNaN(stats.StdDev) {
			t.Errorf("Invalid statistics for tau=%f", tau)
		}
	}

	// Check crossing matrix
	if len(diag.CrossingMatrix) != len(taus) {
		t.Errorf("Expected %d x %d crossing matrix, got %d x %d",
			len(taus), len(taus), len(diag.CrossingMatrix), len(diag.CrossingMatrix))
	}
}

func TestNLRQProcess(t *testing.T) {
	// Create synthetic data from exponential model
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
		F: func(beta []float64, x []float64) float64 {
			return beta[0] * math.Exp(beta[1]*x[0])
		},
		Gradient: func(beta []float64, x []float64) []float64 {
			exp := math.Exp(beta[1] * x[0])
			return []float64{
				exp,
				beta[0] * x[0] * exp,
			}
		},
	}

	// Initial parameter values
	beta0 := []float64{0.5, 0.1}

	// Test multiple quantile levels
	taus := []float64{0.25, 0.5, 0.75}
	fits, err := NLRQProcess(y, x, model, beta0, taus)
	if err != nil {
		t.Fatalf("Failed to fit models: %v", err)
	}

	// Check dimensions
	if fits.N != 5 {
		t.Errorf("Expected 5 observations, got %d", fits.N)
	}

	if fits.P != 2 {
		t.Errorf("Expected 2 parameters, got %d", fits.P)
	}

	if len(fits.Taus) != 3 {
		t.Errorf("Expected 3 quantile levels, got %d", len(fits.Taus))
	}

	// Test predictions
	newX := [][]float64{
		{2.5},
		{3.0},
	}
	predictions, err := fits.Predict(newX)
	if err != nil {
		t.Fatalf("Failed to generate predictions: %v", err)
	}

	if len(predictions) != 3 {
		t.Errorf("Expected predictions for 3 quantiles, got %d", len(predictions))
	}

	// Check that predictions are ordered (no quantile crossing)
	for i := 0; i < len(newX); i++ {
		prev := math.Inf(-1)
		for _, tau := range taus {
			pred := predictions[tau][i]
			if pred < prev {
				t.Errorf("Quantile crossing detected at x=%.1f: tau=%f prediction < previous",
					newX[i][0], tau)
			}
			prev = pred
		}
	}
}

func TestMultiRQFitSummary(t *testing.T) {
	x := [][]float64{
		{1, 0.5},
		{1, 1.0},
		{1, 1.5},
	}
	y := []float64{1.0, 2.0, 2.5}
	taus := []float64{0.25, 0.5, 0.75}

	fits, err := RQProcess(y, x, taus)
	if err != nil {
		t.Fatalf("Failed to fit models: %v", err)
	}

	summary := fits.Summary()
	if summary == "" {
		t.Error("Expected non-empty summary string")
	}
}
