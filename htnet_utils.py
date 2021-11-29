import tensorflow as tf
import numpy as np

def hilbert_tf(x):
    #, N=None, axis=-1):
    """
    Compute the analytic signal, using the Hilbert transform in tensorflow.
    The transformation is done along the last axis by default.
    Adapted from scipy: https://github.com/scipy/scipy/blob/v1.4.1/scipy/signal/signaltools.py#L2012-L2120
    Parameters
    ----------
    x : tensor
        Signal data.  Must be real.
    N : int, optional
        Number of Fourier components.  Default: ``x.shape[axis]``
    axis : int, optional
        Axis along which to do the transformation.  Default: -1.
    Returns
    -------
    xa : ndarray
        Analytic signal of `x`, of each 1-D array along `axis`
    """
    if x.dtype.is_complex:
        raise ValueError("x must be real.")
#     if N is None:
    if tf.__version__[0]=='1':
        N = x.get_shape()[-1].value
    else:
        N = x.get_shape()[-1]
#     if N <= 0:
#         raise ValueError("N must be positive.")
    
    if tf.__version__[0]=='1':
        Xf = tf.spectral.fft(tf.cast(x,dtype=tf.complex64))
    else:
        Xf = tf.signal.fft(tf.cast(x,dtype=tf.complex64))
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2

    if len(x.get_shape().as_list()) > 1:
    #K.ndim(x) > 1:
        ind = [np.newaxis] * len(x.get_shape().as_list()) #K.ndim(x)
        ind[-1] = slice(None)
        h = h[tuple(ind)]
    X_conv = tf.math.multiply(Xf,tf.cast(tf.convert_to_tensor(h),tf.complex64))
    if tf.__version__[0]=='1':
        X_ifft = tf.spectral.ifft(X_conv)
    else:
        X_ifft = tf.signal.ifft(X_conv)
    return X_ifft

def apply_hilbert_tf(x, envelope=False, do_log=False, compute_val='power', data_srate=250):
    """Compute Hilbert transform of signals w/ zero padding in tensorflow.
    Adapted from MNE function
    Parameters
    ----------
    x : tensor, shape (n_times)
        The signal to convert
    n_fft : int
        Size of the FFT to perform, must be at least ``len(x)``.
        The signal will be cut back to original length.
    envelope : bool
        Whether to compute amplitude of the hilbert transform in order
        to return the signal envelope.
    Returns
    -------
    out : array, shape (n_times)
        The hilbert transform of the signal, or the envelope.
    """
#     #Filter data to limit temporal filtering to specific frequencies
#     x = tf.numpy_function(filter_data,[x, 250, 50, 100], Tout=tf.float32)
    
#     from scipy.signal import hilbert
    if tf.__version__[0]=='1':
        n_x = x.get_shape()[-1].value
    else:
        n_x = x.get_shape()[-1]
    #TO DO: remove last timepoint of signal
    hilb_sig = hilbert_tf(x) #hilbert(x, N=n_fft, axis=-1)[..., :n_x] 
    
    if compute_val=='power':
        out = tf.math.abs(hilb_sig)
        if do_log:
            out = tf.math.log1p(out)
    elif compute_val=='phase':
        out = unwrap(angle_custom(hilb_sig)) #tf.math.cos(angle_custom(hilb_sig)) # angle_custom(hilb_sig)
#         tf.print(tf.math.reduce_mean(out))
    elif compute_val=='freqslide':
        ang = angle_custom(hilb_sig) #tf.math.angle(hilb_sig)
        ang = data_srate*diff(unwrap(ang))/(2*np.pi)
        paddings = tf.constant([[0, 0], [0, 0], [0, 0], [0, 1]])
        out = tf.pad(ang, paddings, "CONSTANT") # pad time dimension because of difference function
        # TO DO: apply median filter (use tfa.image.median_filter2d)
#     elif envelope:
#         out = tf.math.abs(hilb_sig)
#         if do_log:
#             out = tf.math.log1p(out)
    return out


def proj_to_roi(in_vals):
    '''
    Project x from channels to ROI using proj mat.
    Parameters
    ----------
    in_vals is a list of 2 tensors:
    
    x : tensor, shape (batch,filter,chans,time)
        The signal to project
    proj_mat : tensor, shape (batch,roi,chans)
        The projection matrix from channels to ROIs
    '''
    x = in_vals[0]
    proj_mat = in_vals[1]
    shape_x = x.get_shape().as_list()

    #Apply projection matrix separately for each filter in x (slow...)
    output_list = []
    for i in range(shape_x[1]):
        output_list.append(proj_mat[:,0,...] @ x[:,i,...])
    x_out = tf.stack(output_list,axis=1)
    return x_out