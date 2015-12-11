%  American Nuclear Society Nuclear Technology Manuscript Template
%  by Katy Huff,
%  based on the template by Seth R. Johnson, 
%  that, in turn, was based on the ANS transaction template and Edward W.
%  Larsen
%
% Usage: \documentclass{ntmanuscript}
% define \title and \author as usual
% Also define \institute for each author
% Finally, for the submitter    
% define \title and \author as usual, and also define \institute and \email
% define \title and \author as usual, and also define \institute and \email

\NeedsTeXFormat{LaTeX2e}

\ProvidesClass{ntmanuscript}[2013/11/16 v1.0 ANS Nuclear Technology Manuscript]

%%%%%%%%%%% TYPE AND GEOMETRY %%%%%%%%%%%
\LoadClass[12pt]{article}

\pagestyle{empty}                                               %no page numbering

\RequirePackage[text={173mm,9in}, columnsep=0.75cm,centering]{geometry}

\RequirePackage[doublespacing]{setspace}

%%use times for math font
\RequirePackage[T1]{fontenc}
\RequirePackage{mathptmx}

%%%%%%%%%%% INCLUDE PACKAGES %%%%%%%%%%%
\RequirePackage{cuted} % for single column in a multicolumn doc
\RequirePackage{indentfirst}    % indent the leading paragraph
\RequirePackage{amsmath}

% for gray line
\RequirePackage{xcolor}
\RequirePackage{titlesec}

% The title may need to be in arial/helvetica
\RequirePackage[scaled]{helvet}

% For nicer tables
\RequirePackage{tabularx}

% For the count of pages, tables, and figures on the title page
\RequirePackage{totcount}

%%%%%%%%%%% REQUIRED VARIABLES %%%%%%%%%%%
\newif\if@institutedefined
\@institutedefinedfalse
\def\@institute{\PackageWarning{ntmanuscript}{No institute has been defined.}}
\newcommand{\institute}[1]{\def\@institute{#1}\@institutedefinedtrue}

\newif\if@emaildefined
\@emaildefinedfalse
\def\@email{\PackageWarning{ntmanuscript}{No email address has been defined.}}
\newcommand{\email}[1]{\def\@email{#1}\@emaildefinedtrue}

\newif\if@keywordsdefined
\@keywordsdefinedfalse
\def\@keywords{\PackageWarning{ntmanuscript}{No keywords have been defined.}}
\newcommand{\keywords}[1]{\def\@keywords{#1}\@keywordsdefinedtrue}

\newif\if@submitterdefined
\@submitterdefinedfalse
\def\@submitter{\PackageWarning{ntmanuscript}{No submitter has been defined.}}
\newcommand{\submitter}[1]{\def\@submitter{#1}\@submitterdefinedtrue}

\newif\if@submitteremaildefined
\@submitteremaildefinedfalse
\def\@submitteremail{\PackageWarning{ntmanuscript}{No submitteremail has been defined.}}
\newcommand{\submitteremail}[1]{\def\@submitteremail{#1}\@submitteremaildefinedtrue}

\newif\if@submitteraddressdefined
\@submitteraddressdefinedfalse
\def\@submitteraddress{\PackageWarning{ntmanuscript}{No submitteraddress has been defined.}}
\newcommand{\submitteraddress}[1]{\def\@submitteraddress{#1}\@submitteraddressdefinedtrue}


%%%%%%%%%%% ABSTRACT %%%%%%%%%%%
\renewenvironment{abstract}{
  \pagebreak[4]
  \noindent\textbf{Abstract}\ignorespaces\par%
  %\end{@twocolumnfalse}
  %\titleline{\color{gray!50!white}\titlerule[10pt]}\medskip}%
}


%%%%%%%%%%% SECTIONING %%%%%%%%%%%
\renewcommand\section{\@startsection%
        {section}{1}{\z@}%              %name, level, indent
        {-\baselineskip}%               %beforeskip
        {.75\baselineskip}%             %afterskip
        {\reset@font\bf\MakeUppercase}% %bold, uppercase
}
\renewcommand{\thesection}{\@Roman\c@section} 

\renewcommand\subsection{\@startsection%
        {subsection}{2}{\z@}%           %name, level, indent
        {-\baselineskip}%               %beforeskip
        {.75\baselineskip}%             %afterskip
        {\reset@font\bf}%               %bold
}
\renewcommand{\thesubsection}{\thesection\@Alph\c@subsection}

\renewcommand\subsubsection{\@startsection
        {subsubsection}{3}{\z@}%        %name, level, indent
        {-.75\baselineskip}%            %beforeskip
        {.5\baselineskip}%              %afterskip
        {\reset@font\emph}%             %italic
}
\renewcommand{\thesubsubsection}{\thesubsection.\@roman\c@subsubsection}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BIBLIOGRAPHY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\renewcommand\@biblabel[1]{#1.} %change from [x] to x.

%% reduce spacing in bibliography; thanks to Jose Carlos Santos.
\let\@oldthebibliography=\thebibliography
\let\@endoldthebibliography=\endthebibliography
\let\@olditemsep=\itemsep
\renewenvironment{thebibliography}[1]{%
        \begin{@oldthebibliography}{#1}%
                \setlength{\itemsep}{-\@olditemsep}%
}%
{%
        \@endoldthebibliography%
        \end{@oldthebibliography}%
}

%%%%%%%%%%% CHANGE TABLE/FIGURE STYLE %%%%%%%%%%%
\renewcommand{\figurename}{Fig.}
\renewcommand{\tablename}{TABLE}

\setlength{\abovecaptionskip}{0.25\baselineskip}
\setlength{\belowcaptionskip}{0.25\baselineskip}
\setlength{\textfloatsep}{0.5\baselineskip}
\setlength{\intextsep}{0.5\baselineskip}

\renewcommand{\bottomfraction}{0.5}

%%%%%%%%%%% GENERATE PORTIONS OF THE DOCUMENT %%%%%%%%%%%

\renewcommand{\sfdefault}{phv}

\renewcommand{\maketitle}{
      %title
    {\sffamily\LARGE\raggedright\@title\par} 
      \vspace{1em}
      %author
      \begin{tabular}[t]{ll}
        \@author
      \end{tabular}
      %institute (optional?)
      \if@institutedefined 
        \par 
        {\let\and\par \itshape \raggedright \@institute} 
      \fi
      %email (optional?)
      \if@emaildefined
        \par 
        {\itshape
        \begin{tabular}[t]{c}
          \@email
        \end{tabular}
        }
      \fi
      %submitter
      \if@submitteraddressdefined 
        \par\bigskip
        \begin{tabularx}{173mm}{lX}
          \textbf{Send proofs to: }&  \@submitter\\
                                   &  \@submitteremail\\
                                   &  \@submitteraddress\\
          \end{tabularx}
      \fi
      %number of pages
      %number of tables
      %number of figures
      \par\bigskip 
      \begin{tabular}{ll}
        \textbf{Number of Pages:}&\regtotcounter{page}\total{page}\\
        \textbf{Number of Tables:}&\regtotcounter{table}\total{table}\\
        \textbf{Number of Figures:}&\regtotcounter{figure}\total{figure}\\
      \end{tabular}
      %keywords
      \if@keywordsdefined 
        \par\bigskip 
        \begin{tabularx}{173mm}{lX}
          \textbf{Keywords: }&\raggedright\@keywords\\ 
        \end{tabularx}
      \fi
}

\AtBeginDocument{
  \maketitle
}

