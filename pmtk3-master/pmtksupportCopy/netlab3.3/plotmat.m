function plotmat(matrix, textcolour, gridcolour, fontsize)
%PLOTMAT Display a matrix.
%
%	Description
%	PLOTMAT(MATRIX, TEXTCOLOUR, GRIDCOLOUR, FONTSIZE) displays the matrix
%	MATRIX on the current figure.  The TEXTCOLOUR and GRIDCOLOUR
%	arguments control the colours of the numbers and grid labels
%	respectively and should follow the usual Matlab specification. The
%	parameter FONTSIZE should be an integer.
%
%	See also
%	CONFFIG, DEMMLP2
%

%	Copyright (c) Ian T Nabney (1996-2001)

[m,n]=size(matrix);
for rowCnt=1:m,
  for colCnt=1:n,
	numberString=num2str(matrix(rowCnt,colCnt));
	text(colCnt-.5,m-rowCnt+.5,numberString, ...
	  'HorizontalAlignment','center', ...
	  'Color', textcolour, ...
	  'FontWeight','bold', ...
	  'FontSize', fontsize);
  end;
end;

set(gca,'Box','on', ...
  'Visible','on', ...
  'xLim',[0 n], ...
  'xGrid','on', ...
  'xTickLabel',[], ...
  'xTick',0:n, ...
  'yGrid','on', ...
  'yLim',[0 m], ...
  'yTickLabel',[], ...
  'yTick',0:m, ...
  'DataAspectRatio',[1, 1, 1], ...
  'GridLineStyle',':', ...
  'LineWidth',3, ...
  'XColor',gridcolour, ...
  'YColor',gridcolour);

end