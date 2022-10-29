# TODO: Remove file, put that in a proper official test
import torch
import pytest



def check(Hi, Wi, Ho, Wo):

    img = torch.randint(0, 256, (1, 3, Hi, Wi), dtype=torch.uint8)
    img = img.contiguous(memory_format=torch.channels_last)

    out = torch.nn.functional.interpolate(img, [Ho, Wo], mode="bilinear", antialias=True)
    out_float = torch.nn.functional.interpolate(img.to(torch.float), [Ho, Wo], mode="bilinear", antialias=True).round().to(torch.uint8)

    # Assert no pixel value differs by more than 1
    torch.testing.assert_allclose(out, out_float, rtol=0, atol=1)


@pytest.mark.parametrize("Hi, Wi, Ho, Wo", (
    (271, 268, 224, 224),
    (256, 128, 512, 256),
    (68, 49, 1549, 2890),
    (10, 15, 512, 320),
    (4, 8, 8, 4),
    (2, 2, 4, 4),
    (10, 15, 10, 15),
))
@pytest.mark.parametrize("reverse", (False, True))
def test_lol(Hi, Wi, Ho, Wo, reverse):
    if reverse:
        Hi, Wi, Ho, Wo = Ho, Wo, Hi, Wi
    
    if Hi == Ho and Wi == Wo:
        pytest.xfail("Segfault lololol")

    check(Hi, Wi, Hi, Wo)  # horizontal
    check(Hi, Wi, Ho, Wi)  # vertical
    check(Hi, Wi, Ho, Wo)  # both
