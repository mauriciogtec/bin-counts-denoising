parse_counts = function(s) {
  # s is the raw string from doq database
  # output is a data frame with cols lower, upper, count
  countdf = s %>% 
    str_sub(1, -4) %>% 
    str_split(",") %>% 
    .[[1]] %>% 
    str_sub(4) %>% 
    str_split(":") %>% 
    map(~ data.frame(
      lower=as.numeric(.[1]),
      upper=as.numeric(str_sub(.[2], 1, -3)),
      count=as.integer(.[3]))) %>% 
    reduce(rbind)
  countdf
}
