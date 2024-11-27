package quantreg

import (
	"fmt"
	"math"
	"sort"

	"github.com/andreasmuller/sparsem"
)

// MultiRQFit represents multiple quantile regression fits
type MultiRQFit struct {
	Fits      map[float64]*RQFit // Map of tau to individual fits
	Taus      []float64          // Sorted list of quantile levels
	N         int                // Number of observations
	P         int                // Number of parameters
	Method    string            // Method used for fitting
	Formula   string            // Model formula
}

// MultiNLRQFit represents multiple non-linear quantile regression fits
type MultiNLRQFit struct {
	Fits      map[float64]*NLRQFit // Map of tau to individual fits
	Taus      []float64            // Sorted list of quantile levels
	N         int                  // Number of observations
	P         int                  // Number of parameters
	Model     NonLinearModel       // The non-linear model
	Formula   string              // Model formula
}

// RQProcess fits multiple quantile regression models
func RQProcess(y []float64, x [][]float64, taus []float64) (*MultiRQFit, error) {
	if len(taus) == 0 {
		return nil, fmt.Errorf("no quantile levels specified")
	}

	// Sort taus for consistent ordering
	sortedTaus := make([]float64, len(taus))
	copy(sortedTaus, taus)
	sort.Float64s(sortedTaus)

	// Check tau values
	for _, tau := range sortedTaus {
		if tau <= 0 || tau >= 1 {
			return nil, fmt.Errorf("tau must be between 0 and 1, got %f", tau)
		}
	}

	fits := make(map[float64]*RQFit)
	var firstFit *RQFit

	// Fit models for each tau
	for _, tau := range sortedTaus {
		fit, err := RQ(y, x, tau)
		if err != nil {
			return nil, fmt.Errorf("failed to fit model for tau=%f: %v", tau, err)
		}
		fits[tau] = fit
		if firstFit == nil {
			firstFit = fit
		}
	}

	return &MultiRQFit{
		Fits:    fits,
		Taus:    sortedTaus,
		N:       firstFit.N,
		P:       firstFit.P,
		Method:  "br",
		Formula: firstFit.Formula,
	}, nil
}

// NLRQProcess fits multiple non-linear quantile regression models
func NLRQProcess(y []float64, x [][]float64, model NonLinearModel, beta0 []float64, taus []float64) (*MultiNLRQFit, error) {
	if len(taus) == 0 {
		return nil, fmt.Errorf("no quantile levels specified")
	}

	// Sort taus for consistent ordering
	sortedTaus := make([]float64, len(taus))
	copy(sortedTaus, taus)
	sort.Float64s(sortedTaus)

	// Check tau values
	for _, tau := range sortedTaus {
		if tau <= 0 || tau >= 1 {
			return nil, fmt.Errorf("tau must be between 0 and 1, got %f", tau)
		}
	}

	fits := make(map[float64]*NLRQFit)
	var firstFit *NLRQFit

	// Fit models for each tau
	for _, tau := range sortedTaus {
		fit, err := NLRQ(y, x, model, beta0, tau)
		if err != nil {
			return nil, fmt.Errorf("failed to fit model for tau=%f: %v", tau, err)
		}
		fits[tau] = fit
		if firstFit == nil {
			firstFit = fit
		}
	}

	return &MultiNLRQFit{
		Fits:    fits,
		Taus:    sortedTaus,
		N:       firstFit.N,
		P:       firstFit.P,
		Model:   model,
		Formula: firstFit.Formula,
	}, nil
}

// Predict generates predictions for all quantile levels
func (m *MultiRQFit) Predict(newX [][]float64) (map[float64][]float64, error) {
	predictions := make(map[float64][]float64)
	
	for _, tau := range m.Taus {
		pred, err := m.Fits[tau].Predict(newX)
		if err != nil {
			return nil, fmt.Errorf("prediction failed for tau=%f: %v", tau, err)
		}
		predictions[tau] = pred
	}
	
	return predictions, nil
}

// Predict generates predictions for all quantile levels
func (m *MultiNLRQFit) Predict(newX [][]float64) (map[float64][]float64, error) {
	predictions := make(map[float64][]float64)
	
	for _, tau := range m.Taus {
		pred, err := m.Fits[tau].Predict(newX)
		if err != nil {
			return nil, fmt.Errorf("prediction failed for tau=%f: %v", tau, err)
		}
		predictions[tau] = pred
	}
	
	return predictions, nil
}

// Summary prints a summary of all fitted models
func (m *MultiRQFit) Summary() string {
	result := fmt.Sprintf("Multiple Quantile Regression\n")
	result += fmt.Sprintf("Number of observations: %d\n", m.N)
	result += fmt.Sprintf("Number of parameters: %d\n", m.P)
	result += fmt.Sprintf("Quantile levels: %v\n\n", m.Taus)

	for _, tau := range m.Taus {
		fit := m.Fits[tau]
		result += fmt.Sprintf("=== Quantile %f ===\n", tau)
		result += "Coefficients:\n"
		for i, coef := range fit.Coefficients {
			result += fmt.Sprintf("  Beta[%d]: %.6f\n", i, coef)
		}
		result += "\n"
	}

	return result
}

// Summary prints a summary of all fitted models
func (m *MultiNLRQFit) Summary() string {
	result := fmt.Sprintf("Multiple Non-linear Quantile Regression\n")
	result += fmt.Sprintf("Number of observations: %d\n", m.N)
	result += fmt.Sprintf("Number of parameters: %d\n", m.P)
	result += fmt.Sprintf("Quantile levels: %v\n\n", m.Taus)

	for _, tau := range m.Taus {
		fit := m.Fits[tau]
		result += fmt.Sprintf("=== Quantile %f ===\n", tau)
		result += "Coefficients:\n"
		for i, coef := range fit.Coefficients {
			result += fmt.Sprintf("  Beta[%d]: %.6f\n", i, coef)
		}
		result += "\n"
	}

	return result
}

// Diagnostics computes various diagnostic measures
type Diagnostics struct {
	PseudoRSquared  float64            // Pseudo R-squared
	ResidualStats   map[float64]Stats  // Residual statistics for each tau
	CrossingMatrix  [][]int            // Matrix showing quantile crossing counts
}

// Stats holds basic statistical measures
type Stats struct {
	Min     float64
	Max     float64
	Median  float64
	Mean    float64
	StdDev  float64
}

// ComputeDiagnostics calculates diagnostic measures for the fits
func (m *MultiRQFit) ComputeDiagnostics() *Diagnostics {
	diag := &Diagnostics{
		ResidualStats:  make(map[float64]Stats),
		CrossingMatrix: make([][]int, len(m.Taus)),
	}

	// Initialize crossing matrix
	for i := range diag.CrossingMatrix {
		diag.CrossingMatrix[i] = make([]int, len(m.Taus))
	}

	// Compute residual statistics for each tau
	for i, tau1 := range m.Taus {
		fit1 := m.Fits[tau1]
		stats := computeStats(fit1.Residuals)
		diag.ResidualStats[tau1] = stats

		// Check for quantile crossings
		for j, tau2 := range m.Taus {
			if j <= i {
				continue
			}
			fit2 := m.Fits[tau2]
			crossings := countCrossings(fit1.Fitted, fit2.Fitted)
			diag.CrossingMatrix[i][j] = crossings
			diag.CrossingMatrix[j][i] = crossings
		}
	}

	// Compute pseudo R-squared using the median fit
	medianFit := m.Fits[0.5]
	if medianFit != nil {
		diag.PseudoRSquared = computePseudoRSquared(medianFit)
	}

	return diag
}

// Helper function to compute basic statistics
func computeStats(data []float64) Stats {
	n := len(data)
	if n == 0 {
		return Stats{}
	}

	sorted := make([]float64, n)
	copy(sorted, data)
	sort.Float64s(sorted)

	var sum, sumSq float64
	for _, v := range data {
		sum += v
		sumSq += v * v
	}
	mean := sum / float64(n)
	variance := (sumSq/float64(n)) - (mean * mean)

	return Stats{
		Min:    sorted[0],
		Max:    sorted[n-1],
		Median: sorted[n/2],
		Mean:   mean,
		StdDev: math.Sqrt(variance),
	}
}

// Helper function to count quantile crossings
func countCrossings(fit1, fit2 []float64) int {
	if len(fit1) != len(fit2) {
		return 0
	}

	crossings := 0
	for i := 1; i < len(fit1); i++ {
		if (fit1[i-1] <= fit2[i-1] && fit1[i] > fit2[i]) ||
			(fit1[i-1] >= fit2[i-1] && fit1[i] < fit2[i]) {
			crossings++
		}
	}
	return crossings
}

// Helper function to compute pseudo R-squared
func computePseudoRSquared(fit *RQFit) float64 {
	var sumRes, sumTot float64
	median := median(fit.Residuals)

	for i, res := range fit.Residuals {
		sumRes += math.Abs(res)
		sumTot += math.Abs(fit.Fitted[i] - median)
	}

	if sumTot == 0 {
		return 0
	}
	return 1 - (sumRes / sumTot)
}

// Helper function to compute median
func median(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}
	sorted := make([]float64, len(data))
	copy(sorted, data)
	sort.Float64s(sorted)
	return sorted[len(sorted)/2]
}
