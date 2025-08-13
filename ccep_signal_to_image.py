# ccep_signal_to_image.py
import numpy as np
import matplotlib.pyplot as plt

def CCEP_signal_to_png(CCEP_signal: np.ndarray, out_path: str, zero_line: bool = True):
    """
    Convert a 1-D CCEP signal into a clean PNG image for computer vision models.
    
    Parameters:
        CCEP_signal (np.ndarray): The 1-D CCEP signal data.
        out_path (str): Output file path for the PNG image.
        zero_line (bool): Whether to draw a horizontal zero reference line.
    """
    # Ensure data is a NumPy float array
    CCEP_signal = np.asarray(CCEP_signal).astype(float)
    
    # Check if input is a valid 1-D signal
    if CCEP_signal.ndim != 1:
        raise ValueError("CCEP_signal must be 1-D")
    
    # Check for invalid values (NaN/Inf)
    if np.isnan(CCEP_signal).any() or np.isinf(CCEP_signal).any():
        raise ValueError("CCEP_signal contains NaN/Inf values")

    # Create figure for plotting
    plt.figure(figsize=(6, 4), dpi=300)

    # Optional horizontal zero line for reference
    if zero_line:
        plt.axhline(0, linewidth=1)

    # Plot the signal
    plt.plot(CCEP_signal, linewidth=1)

    # Remove axes and ticks for a clean image
    plt.axis("off")
    
    # Minimize padding
    plt.tight_layout(pad=0)
    
    # Save image as PNG
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()

if __name__ == "__main__":
    # Example usage: generate a fake CCEP_signal
    x = np.linspace(0, 4*np.pi, 2000)  # Time vector
    CCEP_signal = 0.6*np.sin(2*x) + 0.3*np.sin(7*x)  # Synthetic example
    
    # Save as PNG
    CCEP_signal_to_png(CCEP_signal, "CCEP_signal_example.png")
    print("Saved: CCEP_signal_example.png")
