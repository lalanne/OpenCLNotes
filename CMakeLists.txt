
PROJECT(report NONE)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}")

INCLUDE(UseLuaTex.cmake)
cmake_minimum_required(VERSION 2.4.8)
if(COMMAND cmake_policy)
  cmake_policy(VERSION 2.6)
endif(COMMAND cmake_policy)

SET(FIG_DIR figures)
SET(LATEX_OUTPUT_PATH bin)
SET(CMAKE_CURRENT_SOURCE_DIR src)
SET(LUALATEX_COMPILER_FLAGS "--enable-write18")
ADD_LATEX_DOCUMENT(report.tex
  INPUTS ack.tex introduction.tex  mxm.tex  rows.tex phi.tex conclusions.tex  
		cpu.tex fpga.tex OpenCL.tex  naive.tex  mapping.tex  gpu.tex
		supp.tex rowscols.tex tiling.tex concepts.tex
  PGF_DIRS figures
  IMAGE_DIRS figures
  BIBFILES report.bib
  DEFAULT_PDF
  )
