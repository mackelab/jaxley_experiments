import numpy as np


def sort_spike_number(raster):
    """Sort the raster by the number of spikes"""
    spike_nums = np.sum(raster, axis=1)
    sort_inds = np.flip(np.argsort(spike_nums))
    return raster[sort_inds]


def fft_max_sort(raster):
    """
    Sort the raster with a fast Fourier transform algorithm found here:
    https://medium.com/@ryanberen/sort-binary-vectors-with-fft-33ccc95f5df0
    """
    # Compute the 1d discrete Fourier transform
    RbvFft = np.fft.fft(raster)
    # Find magnitude at each frequency
    RbvFftMagnitudes = np.real(RbvFft * np.conj(RbvFft))
    # Only use a relevant number of frequencies
    FreqsToCheck = RbvFftMagnitudes[
        :, 1 : (1 + min(raster.shape[0] // 2, raster.shape[1] // raster.shape[0]))
    ]
    # Find which frequency is biggest for each vector
    RbvMaxFreqs = np.argmax(FreqsToCheck, 1)
    # Find the phases for each vector's biggest frequency
    RbvPhases = [np.angle(RbvFft[i, RbvMaxFreqs[i] + 1]) for i in range(len(raster))]
    # Damn, Python’s lexsort tool has the absolute worst design.
    RbvSortOrder = np.flipud(np.lexsort((RbvPhases, -RbvMaxFreqs)))
    # Put the vectors into the final clustered & sorted order
    RbvSorted = raster[RbvSortOrder, :]
    # Display the image sideways because that's nicer on the web
    return RbvSorted, RbvSortOrder
