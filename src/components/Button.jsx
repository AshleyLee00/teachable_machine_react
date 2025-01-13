import React from 'react';

const Button = ({ 
  children, 
  onClick, 
  disabled, 
  variant = 'primary',
  size = 'md',
  className = '' 
}) => {
  const baseClasses = 'rounded-md font-medium transition-colors focus:outline-none';
  const variants = {
    primary: 'bg-blue-500 text-white hover:bg-blue-600 disabled:bg-blue-300',
    outline: 'border-2 border-gray-300 hover:border-gray-400 disabled:border-gray-200',
    ghost: 'text-gray-600 hover:bg-gray-100 disabled:text-gray-300'
  };
  const sizes = {
    sm: 'px-2 py-1 text-sm',
    md: 'px-4 py-2',
    lg: 'px-6 py-3 text-lg'
  };

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`${baseClasses} ${variants[variant]} ${sizes[size]} ${className}`}
    >
      {children}
    </button>
  );
};

export default Button;