# quantreg

A Go implementation of quantile regression methods, inspired by the R package quantreg.

## Features

- Linear quantile regression using the Barrodale and Roberts algorithm
- Support for multiple quantile levels (τ)
- Prediction from fitted models
- Model summaries and diagnostics
- Integration with sparse matrix operations via the sparsem package

## Installation

```bash
go get github.com/andreasmuller/quantreg
```

## Usage

```go
package main

import (
    "fmt"
    "github.com/andreasmuller/quantreg"
)

func main() {
    // Prepare your data
    x := [][]float64{
        {1, 0.5},
        {1, 1.0},
        {1, 1.5},
        {1, 2.0},
        {1, 2.5},
    }
    y := []float64{1.0, 2.0, 2.5, 3.0, 4.0}

    // Fit a median regression (τ = 0.5)
    fit, err := quantreg.RQ(y, x, 0.5)
    if err != nil {
        panic(err)
    }

    // Print model summary
    fmt.Println(fit.Summary())

    // Make predictions
    newX := [][]float64{
        {1, 3.0},
        {1, 3.5},
    }
    predictions, err := fit.Predict(newX)
    if err != nil {
        panic(err)
    }
    fmt.Println("Predictions:", predictions)
}
```

## Features to be implemented

1. Non-linear quantile regression
2. Censored quantile regression
3. Bootstrapping methods
4. Additional optimization algorithms
5. More comprehensive model diagnostics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- Koenker, R. (2005) Quantile Regression, Cambridge U. Press
- Koenker, R. et al. (2017) Handbook of Quantile Regression, CRC Press
