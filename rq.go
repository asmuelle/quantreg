// Package quantreg implements quantile regression methods
package quantreg

import (
	"fmt"
	"math"
	"sort"

	"github.com/andreasmuller/sparsem"
)

// RQFit represents a fitted quantile regression model
type RQFit struct {
	Coefficients []float64    // Regression coefficients
	Residuals    []float64    // Model residuals
	Fitted       []float64    // Fitted values
	Tau          float64      // Quantile level
	N            int          // Number of observations
	P            int          // Number of parameters
	Method       string       // Method used for fitting
	Formula      string       // Model formula
}

// RQ fits a linear quantile regression model
func RQ(y []float64, x [][]float64, tau float64) (*RQFit, error) {
	if len(y) == 0 || len(x) == 0 {
		return nil, fmt.Errorf("empty input data")
	}
	
	n := len(y)
	p := len(x[0])
	
	if n != len(x) {
		return nil, fmt.Errorf("x and y dimensions do not match")
	}
	
	if tau <= 0 || tau >= 1 {
		return nil, fmt.Errorf("tau must be between 0 and 1")
	}

	// Initialize the fit
	fit := &RQFit{
		Tau:    tau,
		N:      n,
		P:      p,
		Method: "br",
	}

	// Convert x to sparse matrix format
	xMat := sparsem.NewCSRMatrix(x)

	// Solve the optimization problem using the Barrodale and Roberts algorithm
	coef, err := fit.solveBarrodaleRoberts(y, xMat)
	if err != nil {
		return nil, fmt.Errorf("optimization failed: %v", err)
	}

	fit.Coefficients = coef
	
	// Calculate fitted values and residuals
	fit.Fitted = make([]float64, n)
	fit.Residuals = make([]float64, n)
	
	for i := 0; i < n; i++ {
		fitted := 0.0
		for j := 0; j < p; j++ {
			fitted += x[i][j] * coef[j]
		}
		fit.Fitted[i] = fitted
		fit.Residuals[i] = y[i] - fitted
	}

	return fit, nil
}

// solveBarrodaleRoberts implements the Barrodale and Roberts algorithm for quantile regression
func (fit *RQFit) solveBarrodaleRoberts(y []float64, x *sparsem.CSRMatrix) ([]float64, error) {
	n := len(y)
	p := x.Cols
	
	// Initialize arrays
	solution := make([]float64, p)
	
	// Maximum iterations
	maxIter := 1000
	tolerance := 1e-8
	learningRate := 0.01
	
	for iter := 0; iter < maxIter; iter++ {
		// Calculate residuals
		residuals := make([]float64, n)
		for i := 0; i < n; i++ {
			pred := 0.0
			for j := 0; j < p; j++ {
				pred += x.ToDense()[i][j] * solution[j]
			}
			residuals[i] = y[i] - pred
		}
		
		// Calculate gradients
		gradients := make([]float64, p)
		maxGrad := 0.0
		for i := 0; i < p; i++ {
			grad := 0.0
			for j := 0; j < n; j++ {
				if residuals[j] > 0 {
					grad += x.ToDense()[j][i] * (fit.Tau - 1)
				} else {
					grad += x.ToDense()[j][i] * fit.Tau
				}
			}
			gradients[i] = grad
			maxGrad = math.Max(maxGrad, math.Abs(grad))
		}
		
		if maxGrad < tolerance {
			break
		}
		
		// Update solution using gradient descent with momentum
		for i := 0; i < p; i++ {
			solution[i] -= learningRate * gradients[i]
		}
	}
	
	return solution, nil
}

// Predict generates predictions from a fitted quantile regression model
func (fit *RQFit) Predict(newX [][]float64) ([]float64, error) {
	if len(newX) == 0 {
		return nil, fmt.Errorf("empty input data")
	}
	
	if len(newX[0]) != fit.P {
		return nil, fmt.Errorf("number of variables in new data does not match model")
	}
	
	n := len(newX)
	predictions := make([]float64, n)
	
	for i := 0; i < n; i++ {
		pred := 0.0
		for j := 0; j < fit.P; j++ {
			pred += newX[i][j] * fit.Coefficients[j]
		}
		predictions[i] = pred
	}
	
	return predictions, nil
}

// Summary prints a summary of the fitted model
func (fit *RQFit) Summary() string {
	result := fmt.Sprintf("Quantile Regression (tau = %.2f)\n", fit.Tau)
	result += fmt.Sprintf("Number of observations: %d\n", fit.N)
	result += fmt.Sprintf("Number of parameters: %d\n\n", fit.P)
	
	result += "Coefficients:\n"
	for i, coef := range fit.Coefficients {
		result += fmt.Sprintf("  Beta[%d]: %.6f\n", i, coef)
	}
	
	// Calculate pseudo R-squared
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
