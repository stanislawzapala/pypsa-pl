def add_custom_line_types(network):

    network.line_types.loc["3x468/24-A1F/UHST-261"] = [
        50.0,  # f_nom (Hz)
        0.022,  # r_per_length (Ω/km)
        0.28,  # x_per_length (Ω/km)
        13,  # c_per_length (nF/km)
        2.81458,  # i_nom (kA); equiv. to 1950 MVA per 400 kV circuit
        "ol",  # mounting; ol = overhead line, cs = underground cable
        468,  # cross_section (mm²)
        "PSE-SF.Linia 400kV.0 PL/2024v1 & JAO Static Grid Model v6",  # references
    ]
