% ---------------------------------------------------------------------
% Conference proceedings and article templates for
% personal open-archiving activities
% September 2012
% ---------------------------------------------------------------------

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ipcc}[25/01/2012, v1.0]
\RequirePackage{ifthen}
\RequirePackage{calc}
%\AtEndOfClass{\RequirePackage{microtype}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions*
\LoadClass{article}
\RequirePackage{ifluatex}      % Needed to pick between latex and pdflatex
\RequirePackage{float}

%----------------------------------------------------------------------
%	FONTS
%----------------------------------------------------------------------
\ifluatex
	\usepackage{mathtools}%
	\usepackage[math-style=french,bold-style=upright, nabla=upright,warnings-off={mathtools-colon}]{unicode-math}%
	\usepackage{fontspec}%
	\usepackage{luacode}%

	\fontspec[OpticalSize=15]{New Euler}%
	\setmathfont{New Euler}%
	\setmainfont[ItalicFont={OpenSans Light Italic},BoldFont={OpenSans},BoldItalicFont={OpenSans Italic},Ligatures=TeX]{OpenSans Light}%

	\setsansfont[ItalicFont={OpenSans Light Italic},BoldFont={OpenSans},BoldItalicFont={OpenSans Italic},Ligatures=TeX]{OpenSans Light}%
	\setmonofont{OpenSans}%
	\setmathfont[range=\mathup/{latin,Latin,num}]{OpenSans Light}%
	\setmathfont[range=\mathup/{greek,Greek}]{OpenSans Light}%
	\setmathfont[range=\mathbb/{latin,Latin,num}]{OpenSans }%
	\setmathfont[range=\mathbf/{latin,Latin,num}]{OpenSans }
	\setmathfont[range=\mathbfup/{latin,Latin,num}]{OpenSans }
	\setmathfont[range=\mathbfit/{latin,Latin,num}]{OpenSans  Italic}
	\setmathfont[range=\mathit/{latin,Latin,num}]{OpenSans Light }
	\setmathfont[range=\mathtt/{latin,Latin,num}]{OpenSans }
	\setmathfont[range=\mathrm]{OpenSans Light}
	\setmathfont[range=\mathsfit/{latin,Latin,num}]{OpenSans Light}
	\setmathfont[range=\mathsf/{latin,Latin,num}]{OpenSans Light}
	\setmathfont[range={"002C,"0021,"003A,"003B,"220F,"2206,"2202}]{OpenSans Light}
	\setmathfont[range={"226A,"226B,"00B0,"03F5,"299E,"03D5,"2218,"2221}]{Asana Math}
	\setmathfont[range={\mathcal,\mathbfcal,\mathscr},Scale=1.04]{New Euler}
	\setmathfont[range={"03B1,"03B3,"2211,"220F,"2212,"002B,"2296}]{New Euler}

\else
	\RequirePackage{times}      % Loads the Times-Roman Fonts
	\RequirePackage{mathptmx}   % Loads the Times-Roman Math Fonts
	\RequirePackage[utf8]{inputenc}
  \AtEndOfClass{\RequirePackage{microtype}}
	\RequirePackage{amsmath,amsfonts,amssymb}
\fi
%----------------------------------------------------------------------
%	VARIOUS USEFUL PACKAGES
%----------------------------------------------------------------------

\RequirePackage{graphicx}
\RequirePackage[svgnames,cmyk]{xcolor}
\RequirePackage[english]{babel}
\RequirePackage{booktabs}
\usepackage{minted}

%----------------------------------------------------------------------
%	MARGINS	
%----------------------------------------------------------------------
	  	  
\RequirePackage[left=2cm,%
                right=2cm,%
				top=2.25cm,%
				bottom=2.25cm,%
				headheight=15pt,%
				a4paper]{geometry}%
				
%----------------------------------------------------------------------
%	FIGURES AND TABLES CAPTIONS
%----------------------------------------------------------------------

\RequirePackage[labelfont={color=IrishGreen,bf,sf,small},%
                labelsep=period,%
                justification=raggedright,font=small]{caption}
\setlength{\abovecaptionskip}{0pt}
\setlength{\belowcaptionskip}{0pt}

				
%----------------------------------------------------------------------
%	PAGE HEADER
%----------------------------------------------------------------------

\RequirePackage{fancyhdr}  % Needed to define custom headers/footers
\RequirePackage{lastpage}  % Number of pages in the document
\pagestyle{fancy}          % Enables the custom headers/footers
\renewcommand{\headrulewidth}{0.5mm}
\renewcommand{\headrule}{
  \kern-0.6mm
  {\color{IrishGreen}\hrule height \headrulewidth\hfill}
  \kern 0.6mm
  }%
% Headers
\lhead{}%
\chead{}%
\rhead{\color{IrishGreen}\small\sffamily\bfseries\@PaperTitle\  --- \thepage/\pageref{LastPage}}
% Footers
\lfoot{}%
\cfoot{}%
\rfoot{}%
\renewcommand{\footrulewidth}{0.0mm}% % No footer rule

\renewcommand{\footnoterule}{%
  \kern -3pt
  {\color{IrishGreen}\hrule width 6.24cm height 1pt}
  \kern 2pt
}

%----------------------------------------------------------------------
%	SECTION/SUBSECTION/PARAGRAPH SET-UP
%----------------------------------------------------------------------

\RequirePackage[explicit]{titlesec}
\titleformat{\section}
  {\color{IrishGreen}\large\sffamily\bfseries}
  {}
  {0em}
  {\colorbox{RoyalBlue}{\parbox{\dimexpr\linewidth-2\fboxsep\relax}{\centering\arabic{section}. #1}}}
  []
\titleformat{name=\section,numberless}
  {\color{IrishGreen}\large\sffamily\bfseries}
  {}
  {0em}
  {\colorbox{RoyalBlue}{\parbox{\dimexpr\linewidth-2\fboxsep\relax}{\centering#1}}}
  []  
\titleformat{\subsection}
  {\color{IrishGreen}\sffamily\bfseries}
  {\thesubsection}
  {0.5em}
  {#1}
  []
\titleformat{\subsubsection}
  {\color{IrishGreen}\sffamily\small\bfseries}
  {\thesubsubsection}
  {0.5em}
  {#1}
  []    
\titleformat{\paragraph}[runin]
  {\sffamily\small\bfseries}
  {}
  {0em}
  {#1} 
\titlespacing*{\section}{0pc}{3ex \@plus4pt \@minus3pt}{5pt}
\titlespacing*{\subsection}{0pc}{2.5ex \@plus3pt \@minus2pt}{0pt}
\titlespacing*{\subsubsection}{0pc}{2ex \@plus2.5pt \@minus1.5pt}{0pt}
\titlespacing*{\paragraph}{0pc}{1.5ex \@plus2pt \@minus1pt}{10pt}

%----------------------------------------------------------------------
%	TABLEOFCONTENTS SET-UP
%----------------------------------------------------------------------
\newlength{\tocsep} 
\setlength\tocsep{1.5pc} % Sets the indentation of the sections in the table of contents
\setcounter{tocdepth}{3} % Three levels in the table of contents section: sections, subsections and subsubsections
%
\setlength{\parskip}{2mm}

\usepackage{titletoc}
\contentsmargin{0cm}
\titlecontents{section}[\tocsep]
  {\addvspace{4pt}\small\bfseries\sffamily}
  {\color{IrishGreen}\contentslabel[\thecontentslabel]{\tocsep}}
  {}
  {\hfill\color{IrishGreen}\thecontentspage}
  []
\titlecontents{subsection}[\tocsep]
  {\addvspace{2pt}\sffamily}
  {\color{IrishGreen}\contentslabel[\thecontentslabel]{\tocsep}}
  {}
  {\ \titlerule*[.5pc]{.}\ \color{IrishGreen}\thecontentspage}
  []
\titlecontents*{subsubsection}[\tocsep]
  {\footnotesize\sffamily}
  {}
  {}
  {}
  [\ \textbullet\ ]  

%----------------------------------------------------------------------
%	MULTIPLE AUTHOR SET
%----------------------------------------------------------------------  
  
\newcount\@authcnt
\newcount\@tmpcnt\@tmpcnt\z@

\def\@affiliation{%
  \ifnum\@tmpcnt<\@authcnt
   \global\advance\@tmpcnt1
    \raggedright \csname @auth\romannumeral\the\@tmpcnt\endcsname\hfill\\%
   \let\next\@affiliation \vskip1pt
  \else
   \let\next\relax
  \fi
\next}
	 
\newcommand{\affiliation}[1]{%
    \global\advance\@authcnt1
    \expandafter\gdef\csname @auth\romannumeral\the\@authcnt\endcsname
    {#1}}

	
%----------------------------------------------------------------------
%	LIST CONTROL
%----------------------------------------------------------------------  

\RequirePackage{enumitem}
%\setlist{nolistsep} % Uncomment to remove spacing between items in lists (enumerate, itemize)

%----------------------------------------------------------------------
%	ABSTRACT+AUTHOR FRAME
%----------------------------------------------------------------------  

\newcommand{\PaperTitle}[1]{\def\@PaperTitle{#1}}
\newcommand{\Archive}[1]{\def\@Archive{#1}}
\newcommand{\Authors}[1]{\def\@Authors{#1}}
\newcommand{\JournalInfo}[1]{\def\@JournalInfo{#1}}
\newcommand{\Abstract}[1]{\def\@Abstract{#1}}
\newcommand{\Keywords}[1]{\def\@Keywords{#1}}

% ---------------------------------------------------------------------

\renewcommand{\@maketitle}{%
\onecolumn{%
\thispagestyle{empty}%
\vskip-36pt%
{\raggedleft\small\sffamily\bfseries\@JournalInfo\\\@Archive\par}%
\vskip20pt%
{\raggedright\color{IrishGreen}\sffamily\bfseries\fontsize{20}{25}\selectfont \@PaperTitle\par}%
\vskip10pt%
{\raggedright\color{IrishGreen}\sffamily\fontsize{12}{16}\selectfont  \@Authors\par}%
\vskip18pt%
\fcolorbox{IrishGreen}{white}{%
\parbox{\textwidth-2\fboxsep-2\fboxrule}{\centering%
\colorbox{RoyalBlue}{%
\parbox{\textwidth-4\fboxsep-2\fboxrule}{%
\ifx\@Keywords\@empty%
\sffamily\textbf{\abstractname}\\\@Abstract%
\else%
\sffamily\textbf{\abstractname}\\\@Abstract\\[4pt]%
\textbf{\keywordname}\\\@Keywords%
\fi%
}%
}%
\vskip4pt%
\begingroup%
\raggedright\sffamily\small%
\footnotesize\@affiliation\par%
\endgroup%%
}%
}%
\vskip25pt%
}%
}%



%----------------------------------------------------------------------
%	Shortcuts for citations
%----------------------------------------------------------------------  

\DeclareRobustCommand{\jref}[1]{\ref{#1}}
\DeclareRobustCommand{\gref}[1]{(\jref{#1})}
\DeclareRobustCommand{\eref}[1]{eq. (\jref{#1})}
\DeclareRobustCommand{\fref}[1]{fig. \jref{#1}}
\DeclareRobustCommand{\tref}[1]{table \jref{#1}}
\DeclareRobustCommand{\sref}[1]{section \jref{#1}}
\DeclareRobustCommand{\cref}[1]{chapter \jref{#1}}
\DeclareRobustCommand{\aref}[1]{algorithm \jref{#1}}
\DeclareRobustCommand{\appref}[1]{appendix \jref{#1}}
\DeclareRobustCommand{\rref}[1]{ref. \citep{#1}}
\DeclareRobustCommand{\Eref}[1]{Eq. (\jref{#1})}
\DeclareRobustCommand{\Fref}[1]{Fig. \jref{#1}}
\DeclareRobustCommand{\Tref}[1]{Table \jref{#1}}
\DeclareRobustCommand{\Sref}[1]{Section \jref{#1}}
\DeclareRobustCommand{\Cref}[1]{Chapter \jref{#1}}
\DeclareRobustCommand{\Aref}[1]{Algorithm \jref{#1}}
\DeclareRobustCommand{\Appref}[1]{Appendix \jref{#1}}
\DeclareRobustCommand{\Rref}[1]{Ref. \citep{#1}}

%----------------------------------------------------------------------
%	Boxes for code
%----------------------------------------------------------------------  

\newminted{fortran}{bgcolor=RoyalBlue,
              linenos=true,
              frame=lines,
              showspaces=false,
              fontsize=\footnotesize,
              numbersep=2pt,
              xleftmargin=6pt,
              %fontsize=\theFancyVerbLine
}


%----------------------------------------------------------------------
%	Floats
%----------------------------------------------------------------------  
\newfloat{code}{thp}{lop}
\floatname{code}{Code}


\newfloat{algo}{thp}{lop}
\floatname{algo}{Algorithm}

%----------------------------------------------------------------------
%	REFERENCES
%----------------------------------------------------------------------  

% Remove brackets from numbering in List of References
\renewcommand{\@biblabel}[1]{\bfseries\color{IrishGreen}\textsuperscript{[#1]}}
%\setlength{\bibitemsep}{0cm}
\let\oldbibliography\thebibliography
\renewcommand{\thebibliography}[1]{%
\addcontentsline{toc}{section}{\refname}%
\oldbibliography{#1}%
\setlength\itemsep{0pt}}%


\ifluatex
\def\readFileCols#1#2{%
  \directlua{%
   local input = io.open('#1', 'r')
   dataTable = {}
	 k=0
   for line in input:lines() do
		 if k>0 then 
       local split = string.explode(line,'#2')
       tableItem={split[1]}%item name is the first input argument of the line
       tableItem.arg1=split[1]
       tableItem.arg2=split[2]
       tableItem.arg3=split[3]
       tableItem.arg4=split[4]
       tableItem.arg5=split[5]
       tableItem.arg6=split[6]
       tableItem.arg7=split[7]
       tableItem.arg8=split[8]
       tableItem.arg9=split[9]
       table.insert(dataTable,tableItem)
			end
		 k=k+1
    end
    table.sort(dataTable, function(a,b) return tonumber(a.arg1) < tonumber(b.arg1) end)
    input:close()
  }%
}
\def\readFileFiveCols#1#2{%
  \directlua{%
   local input = io.open('#1', 'r')
   dataTable = {}
	 k=0
   for line in input:lines() do
		 if k>0 then 
       local split = string.explode(line,'#2')
       tableItem={split[1]}%item name is the first input argument of the line
       tableItem.arg1=split[1]
       tableItem.arg2=split[2]
       tableItem.arg3=split[3]
       tableItem.arg4=split[4]
       tableItem.arg5=split[5]
       table.insert(dataTable,tableItem)
			end
		 k=k+1
    end
    table.sort(dataTable, function(a,b) return tonumber(a.arg1) < tonumber(b.arg1) end)
    input:close()
  }%
}
\fi
