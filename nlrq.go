package quantreg

import (
	"fmt"
	"math"
	"sort"
)

// NonLinearModel represents a non-linear model function and its gradient
type NonLinearModel struct {
	// F is the model function that takes parameters β and predictors x and returns the predicted value
	F func(beta []float64, x []float64) float64

	// Gradient is the gradient of F with respect to β
	// Returns partial derivatives ∂F/∂β for each parameter
	Gradient func(beta []float64, x []float64) []float64
}

// NLRQFit represents a fitted non-linear quantile regression model
type NLRQFit struct {
	Coefficients []float64      // Model parameters β
	Residuals    []float64      // Model residuals
	Fitted       []float64      // Fitted values
	Tau          float64        // Quantile level
	N            int            // Number of observations
	P            int            // Number of parameters
	Model        NonLinearModel // The non-linear model
	Formula      string         // Model formula
}

// NLRQ fits a non-linear quantile regression model
func NLRQ(y []float64, x [][]float64, model NonLinearModel, beta0 []float64, tau float64) (*NLRQFit, error) {
	if len(y) == 0 || len(x) == 0 {
		return nil, fmt.Errorf("empty input data")
	}

	n := len(y)
	p := len(beta0)

	if n != len(x) {
		return nil, fmt.Errorf("x and y dimensions do not match: len(y)=%d, len(x)=%d", len(y), len(x))
	}

	if tau <= 0 || tau >= 1 {
		return nil, fmt.Errorf("tau must be between 0 and 1")
	}

	// Initialize the fit
	fit := &NLRQFit{
		Tau:   tau,
		N:     n,
		P:     p,
		Model: model,
	}

	// Optimize parameters using interior point method
	coef, err := fit.solveInteriorPoint(y, x, beta0)
	if err != nil {
		return nil, fmt.Errorf("optimization failed: %v", err)
	}

	fit.Coefficients = coef

	// Calculate fitted values and residuals
	fit.Fitted = make([]float64, n)
	fit.Residuals = make([]float64, n)

	for i := 0; i < n; i++ {
		fitted := model.F(coef, x[i])
		fit.Fitted[i] = fitted
		fit.Residuals[i] = y[i] - fitted
	}

	return fit, nil
}

// solveInteriorPoint implements the interior point method for non-linear quantile regression
func (fit *NLRQFit) solveInteriorPoint(y []float64, x [][]float64, beta0 []float64) ([]float64, error) {
	n := len(y)
	p := len(beta0)

	// Algorithm parameters
	maxIter := 1000
	tolerance := 1e-8
	learningRate := 0.01
	t := 1.0 // Barrier parameter
	mu := 10.0 // Barrier update parameter

	// Initialize solution with starting values
	beta := make([]float64, p)
	copy(beta, beta0)

	for iter := 0; iter < maxIter; iter++ {
		// Calculate residuals and their gradients
		residuals := make([]float64, n)
		gradients := make([][]float64, n)

		for i := 0; i < n; i++ {
			predicted := fit.Model.F(beta, x[i])
			residuals[i] = y[i] - predicted
			gradients[i] = fit.Model.Gradient(beta, x[i])
		}

		// Calculate the objective gradient
		objGrad := make([]float64, p)
		for j := 0; j < p; j++ {
			for i := 0; i < n; i++ {
				if residuals[i] > 0 {
					objGrad[j] += gradients[i][j] * (fit.Tau - 1)
				} else {
					objGrad[j] += gradients[i][j] * fit.Tau
				}
			}
		}

		// Check convergence
		maxGrad := 0.0
		for j := 0; j < p; j++ {
			maxGrad = math.Max(maxGrad, math.Abs(objGrad[j]))
		}

		if maxGrad < tolerance {
			break
		}

		// Update parameters
		for j := 0; j < p; j++ {
			beta[j] -= learningRate * objGrad[j]
		}

		// Update barrier parameter
		t *= mu
	}

	return beta, nil
}

// Predict generates predictions from a fitted non-linear quantile regression model
func (fit *NLRQFit) Predict(newX [][]float64) ([]float64, error) {
	if len(newX) == 0 {
		return nil, fmt.Errorf("empty input data")
	}

	n := len(newX)
	predictions := make([]float64, n)

	for i := 0; i < n; i++ {
		predictions[i] = fit.Model.F(fit.Coefficients, newX[i])
	}

	return predictions, nil
}

// Summary prints a summary of the fitted non-linear model
func (fit *NLRQFit) Summary() string {
	result := fmt.Sprintf("Non-linear Quantile Regression (tau = %.2f)\n", fit.Tau)
	result += fmt.Sprintf("Number of observations: %d\n", fit.N)
	result += fmt.Sprintf("Number of parameters: %d\n\n", fit.P)

	result += "Coefficients:\n"
	for i, coef := range fit.Coefficients {
		result += fmt.Sprintf("  Beta[%d]: %.6f\n", i, coef)
	}

	// Calculate residual statistics
	sortedResiduals := make([]float64, len(fit.Residuals))
	copy(sortedResiduals, fit.Residuals)
	sort.Float64s(sortedResiduals)

	var sumAbs float64
	for _, r := range sortedResiduals {
		sumAbs += math.Abs(r)
	}

	result += fmt.Sprintf("\nResidual summary:\n")
	result += fmt.Sprintf("  Min: %.6f\n", sortedResiduals[0])
	result += fmt.Sprintf("  Max: %.6f\n", sortedResiduals[len(sortedResiduals)-1])
	result += fmt.Sprintf("  Median: %.6f\n", sortedResiduals[len(sortedResiduals)/2])

	return result
}
