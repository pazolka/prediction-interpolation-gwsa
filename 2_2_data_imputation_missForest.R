library(missForest)
library(doParallel)

# read data
df = read.csv("./data/Bangladesh/target/filtered_gws_ts_data_1961_2019.csv",
              row.names="Date", check.names = FALSE)

# create a mask for leading and trailing nans
mask_lead_trail_na <- function(x) {
  n <- length(x)
  nz <- which(!is.na(x))
  if (length(nz) == 0L) return(rep(TRUE, n))  # all NA
  first_valid <- nz[1]; last_valid <- nz[length(nz)]
  m <- rep(FALSE, n)
  if (first_valid > 1) m[1:(first_valid - 1)] <- TRUE
  if (last_valid < n)  m[(last_valid + 1):n] <- TRUE
  m
}

mask_mat <- sapply(df, mask_lead_trail_na)
mode(mask_mat) <- "logical"                       # ensure logical

# impute with missForest
doParallel::registerDoParallel(cores = 4) # set based on number of CPU cores
doRNG::registerDoRNG(seed = 123)
fit <- missForest(df, parallelize = 'forests', maxiter = 10, ntree = 200, verbose = TRUE ) # variablewise = TRUE for ts-wise oob to see which ts are harder to impute

# Get the imputed data and the OOB error estimate
imputed_df <- fit$ximp
oob_error   <- fit$OOBerror

# trim time series post-imputation
imputed_df[mask_mat] <- NA

# write data
write.csv(cbind(Date = rownames(imputed_df), imputed_df), 
          "./data/Bangladesh/target/filtered_filled_missForest_gws_ts_data_1961_2019.csv",
          row.names = FALSE)

