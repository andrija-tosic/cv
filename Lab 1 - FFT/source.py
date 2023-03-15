import numpy as np
import cv2
import matplotlib.pyplot as plt


def normalize_uint8(input: np.ndarray) -> cv2.Mat:
    return cv2.normalize(input, None, alpha=img_min,
                         beta=img_max, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)


if __name__ == '__main__':
    img = cv2.imread('input.png', cv2.IMREAD_GRAYSCALE)

    e1 = cv2.getTickCount()

    img_min, img_max = np.amin(img, (0, 1)), np.amax(img, (0, 1))

    # Po OpenCV dokumentaciji, cv2.dft je brzi od numpy.fft.fft2
    dft = cv2.dft(img.astype('float32'), flags=cv2.DFT_COMPLEX_OUTPUT)

    dft_shift = np.fft.fftshift(dft)

    mag, phase = cv2.cartToPolar(dft_shift[:, :, 0], dft_shift[:, :, 1])

    mag_spec = np.log(mag) / 20

    # Mapiranje spektra u opseg [0, 1]
    mask = cv2.normalize(mag_spec, None, alpha=0, beta=1,
                         norm_type=cv2.NORM_MINMAX)

    # Mapiranje maske u skup {0, 1}
    mask = cv2.threshold(mask, 0.8, 1, cv2.THRESH_BINARY)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    print('kernel\n', kernel)

    # Nema potrebe za dilaciju maske jer je 1 piksel uzrok šuma
    # mask = cv2.dilate(mask, kernel, iterations=1)

    c_x = mask.shape[1] // 2
    c_y = mask.shape[0] // 2

    # Kruznica u centru odbacuje belinu
    mask = cv2.circle(mask, (c_x, c_y), radius=10,
                      color=0, thickness=cv2.FILLED)

    mask = 1 - mask

    # Magnituda se postavlja na 0 tamo gde su tacke (gde maska nije 0)
    mag_filtered = np.log(mag)/20 * mask
    mag *= mask

    # Povratak u prostorni domen
    real, imag = cv2.polarToCart(mag, phase)
    complex_spec = cv2.merge([real, imag])
    f_ishift = np.fft.ifftshift(complex_spec)
    f_idft = cv2.idft(f_ishift)
    f_idft = cv2.magnitude(f_idft[:, :, 0], f_idft[:, :, 1])

    denoised_norm = normalize_uint8(f_idft)
    mag_filtered_norm = normalize_uint8(mag_filtered)
    spec_norm = normalize_uint8(mag_spec)

    e2 = cv2.getTickCount()

    print('time: ', (e2-e1) / cv2.getTickFrequency() * 1000, 'ms')

    _figure, axes_arr = plt.subplots(2, 3)
    axes_arr[0][0].set_title('Original image')
    axes_arr[0][0].imshow(img, cmap='gray')

    axes_arr[0][1].set_title('Magnitude spectrum')
    axes_arr[0][1].imshow(mag_spec, cmap='gray')

    axes_arr[0][2].set_title('Phase spectrum')
    axes_arr[0][2].imshow(phase, cmap='gray')

    axes_arr[1][0].set_title('Mask')
    axes_arr[1][0].imshow(mask, cmap='gray')

    axes_arr[1][1].set_title('Masked magnitude spectrum')
    axes_arr[1][1].imshow(mag_filtered_norm, cmap='gray')

    axes_arr[1][2].set_title('Denoised result')
    axes_arr[1][2].imshow(denoised_norm, cmap='gray')

    cv2.imwrite("fft_mag.png", spec_norm)
    cv2.imwrite("fft_mag_filtered.png", mag_filtered_norm)
    cv2.imwrite("output.png", denoised_norm)

    plt.show()
