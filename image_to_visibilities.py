import numpy as np
import matplotlib.pyplot as plt

from numpy.fft import fftfreq, fftshift, fftn, ifftshift


def image_to_visibilities_2D(img, X, Y, N):
    """
    This function takes an input 2D image domain array, and returns the 2D visibilities for that image.
    """
    # Order of Fourier operations:
    # 1) Pad -: Input array should already be padded.
    # 2) Shift -: fftshift-roll-roll necessary due to off by one error.
    # 3) FT -: fft
    # 4) Inverse shift -: ifftshift
    Vis = ifftshift(fftn(np.roll(np.roll(fftshift(img), 1, axis=0), 1, axis=1)))
    # Creating the Fourier grid:
    # N is number of sample points
    # T is sample spacing
    u_vec = fftfreq(N, X / N)
    v_vec = fftfreq(N, Y / N)
    # Creating the u and v plane:
    u_arr, v_arr = np.meshgrid(u_vec, v_vec)

    return u_arr, v_arr, Vis


"""It isn't perfect and it doesn't convert things into proper units, but it should do the FFT correctly. 
The inputs img, X, and Y are 2D arrays. Where X and Y define your coordinate grid, which for radio 
astronomy are usually defined with the coordinates (l,m).  X and Y are useful because the are used 
to create the (u,v) coordinate grid. The input 'img' is the sky model, this is usually a zero padded
 image."""


if __name__ == "__main__":
    img = np.random.rand(100, 100)
    padded_img = np.pad(img, pad_width=(1, 1))

    xx = np.linspace(0, 112, 112)
    yy = np.linspace(0, 112, 112)

    x, y = np.meshgrid(xx, yy)
    N = 112

    u_arr, v_arr, vis = image_to_visibilities_2D(img, x.size, y.size, N)

    plt.imshow(img)
    plt.savefig("img.png")

    plt.imshow(np.angle(vis))
    plt.savefig("vis_amplitude.png")

    plt.plot(u_arr, v_arr, marker=".", color="k", linestyle="none")
    plt.savefig("uv_arrs.png")
