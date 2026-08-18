# ==============================================================================
# 00_download_and_ingest.R
# Automated downloading, decompression and cleaning of data structures
# ==============================================================================

library(httr)
options(timeout = 3600) #Global R timeout setting (1 hour = 3,600 seconds)

# 1. Directory configuration
raw_dir <- "data/raw"
if (!dir.exists(raw_dir)) dir.create(raw_dir, recursive = TRUE)

# 2. Backblaze’s official Q1 2024 URL
zip_url  <- "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_Q1_2024.zip"
zip_path <- file.path(raw_dir, "data_Q1_2024.zip")

# 3. Download if the archive does not exist or is incomplete
if (!file.exists(zip_path) || file.info(zip_path)$size < 100000000) {
  message("--> Download the Backblaze Q1 2024 archive (please wait)...")
  #Force the use of the libcurl/curl engine to bypass the RStudio driver, which sets the timeout to 60 seconds
  download.file(
    url      = zip_url, 
    destfile = zip_path, 
    method   = if (capabilities("libcurl")) "libcurl" else "auto", 
    mode     = "wb"
  )
} else {
  message("--> Zip archive already present locally.")
}

# 4. Extracting the archive
message("--> Extract the zip archive to data/raw/...")
unzip(zip_path, exdir = raw_dir)

# 5. Resolving the issue of unwanted macOS subfolders and files
message("--> Standardisation of the structure of CSV files...")

# A. Immediate deletion of the __MACOSX junk folder created by macOS
macosx_dir <- file.path(raw_dir, "__MACOSX")
if (dir.exists(macosx_dir)) {
  unlink(macosx_dir, recursive = TRUE)
  message("   [OK] The __MACOSX folder has been deleted.")
}

# B. Migrate the CSV files from the extracted subfolder to data/raw/
nested_dir <- file.path(raw_dir, "data_Q1_2024")
if (dir.exists(nested_dir)) {
  nested_csvs <- list.files(nested_dir, pattern = "\\.csv$", full.names = TRUE)
  if (length(nested_csvs) > 0) {
    file.rename(nested_csvs, file.path(raw_dir, basename(nested_csvs)))
    message(sprintf("   [OK] %d CSV files moved from the subfolder to %s/", length(nested_csvs), raw_dir))
  }
  # Deleting the empty folder
  unlink(nested_dir, recursive = TRUE)
}

# C. Cleaning up Apple’s hidden files (ex: ._2024-01-01.csv)
apple_dot_files <- list.files(raw_dir, pattern = "^\\._.*\\.csv$", full.names = TRUE)
if (length(apple_dot_files) > 0) {
  file.remove(apple_dot_files)
  message(sprintf("   [OK] %d malicious Apple cache files (._*.csv) deleted.", length(apple_dot_files)))
}

# 6. Final check
valid_csvs <- list.files(raw_dir, pattern = "^2024-.*\\.csv$", full.names = TRUE)
message(sprintf("==> Import completed successfully: %d valid CSV files ready for processing.", length(valid_csvs)))