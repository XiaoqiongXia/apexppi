from apexppi import ApexPPI, ApexPPIPredictor, __version__


def test_public_symbols_are_exported():
    assert ApexPPI.__name__ == "ApexPPI"
    assert ApexPPIPredictor.__name__ == "ApexPPIPredictor"
    assert __version__ == "0.1.0"
