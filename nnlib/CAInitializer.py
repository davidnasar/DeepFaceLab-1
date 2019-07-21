import numpy as np

def _compute_fans(shape, data_format='channels_last'):
    """Computes the number of input and output units for a weight shape.
    # Arguments
        shape: Integer shape tuple.
        data_format: Image data format to use for convolution kernels.
            Note that all kernels in Keras are standardized on the
            `channels_last` ordering (even when inputs are set
            to `channels_first`).
    # Returns
        A tuple of scalars, `(fan_in, fan_out)`.
    # Raises
        ValueError: in case of invalid `data_format` argument.
    """
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) in {3, 4, 5}:
        # Assuming convolution kernels (1D, 2D or 3D).
        # TH kernel shape: (depth, input_depth, ...)
        # TF kernel shape: (..., input_depth, depth)
        if data_format == 'channels_first':
            receptive_field_size = np.prod(shape[2:])
            fan_in = shape[1] * receptive_field_size
            fan_out = shape[0] * receptive_field_size
        elif data_format == 'channels_last':
            receptive_field_size = np.prod(shape[:-2])
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
        else:
            raise ValueError('Invalid data_format: ' + data_format)
    else:
        # No specific assumptions.
        fan_in = np.sqrt(np.prod(shape))
        fan_out = np.sqrt(np.prod(shape))
    return fan_in, fan_out

def _create_basis(filters, size, floatx, eps_std):
    if size == 1:
        return np.random.normal(0.0, eps_std, (filters, size))

    nbb = filters // size + 1
    li = []
    for i in range(nbb):
        a = np.random.normal(0.0, 1.0, (size, size))
        a = _symmetrize(a)
        u, _, v = np.linalg.svd(a)
        li.extend(u.T.tolist())
    p = np.array(li[:filters], dtype=floatx)
    return p

def _symmetrize(a):
    return a + a.T - np.diag(a.diagonal())

def _scale_filters(filters, variance):
    c_var = np.var(filters)
    p = np.sqrt(variance / c_var)
    return filters * p

def CAGenerateWeights ( shape, floatx, data_format, eps_std=0.05, seed=None ):
    if seed is not None:
        np.random.seed(seed)

    fan_in, fan_out = _compute_fans(shape, data_format)
    variance = 2 / fan_in

    rank = len(shape)
    if rank == 3:
        row, stack_size, filters_size = shape

        transpose_dimensions = (2, 1, 0)
        kernel_shape = (row,)
        correct_ifft = lambda shape, s=[None]: np.fft.irfft(shape, s[0])
        correct_fft = np.fft.rfft

    elif rank == 4:
        row, column, stack_size, filters_size = shape

        transpose_dimensions = (2, 3, 1, 0)
        kernel_shape = (row, column)
        correct_ifft = np.fft.irfft2
        correct_fft = np.fft.rfft2

    elif rank == 5:
        x, y, z, stack_size, filters_size = shape

        transpose_dimensions = (3, 4, 0, 1, 2)
        kernel_shape = (x, y, z)
        correct_fft = np.fft.rfftn
        correct_ifft = np.fft.irfftn
    else:
        raise ValueError('rank unsupported')

    kernel_fourier_shape = correct_fft(np.zeros(kernel_shape)).shape

    init = []
    for i in range(filters_size):
        basis = _create_basis(
            stack_size, np.prod(kernel_fourier_shape), floatx, eps_std)
        basis = basis.reshape((stack_size,) + kernel_fourier_shape)

        filters = [correct_ifft(x, kernel_shape) +
                   np.random.normal(0, eps_std, kernel_shape) for
                   x in basis]

        init.append(filters)

    # Format of array is now: filters, stack, row, column
    init = np.array(init)
    init = _scale_filters(init, variance)
    return init.transpose(transpose_dimensions)
