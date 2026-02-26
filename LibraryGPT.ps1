if ($env:COMPUTERNAME -eq "GRAPHENE") {
    cd C:/Users/zsg/Dropbox/Databases/LibraryGPT
} else {
    r:
    cd r:/Dropbox/Databases/LibraryGPT
}

conda activate streamlit311

streamlit run LibraryGPT.py
